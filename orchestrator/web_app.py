"""FastAPI service for the chess orchestrator camera-driven move flow."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

import chess
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable
from pydantic import BaseModel, Field
import yaml

from orchestrator.camera import DirectoryCamera
from orchestrator.chess_types import DifficultyConfig
from orchestrator.difficulty import DifficultyController
from orchestrator.engine_service import StockfishService
from orchestrator.executor import build_executor
from orchestrator.game_logger import ChessMoveLogger
from orchestrator.game_service import ChessGameService
from orchestrator.game_state import ChessMemoryStore
from orchestrator.policy_agent import ChessOrchestratorAgent
from orchestrator.vision_agent import VisionModelResolver, VisionProviderSettings

LOGGER = logging.getLogger(__name__)


class AnalyseRequest(BaseModel):
    """Payload for analysing a player's completed move."""

    player_time_s: Optional[float] = None
    image_path: Optional[str] = None
    analysis_image_data_url: Optional[str] = None
    ground_truth_piece_placement: Optional[str] = None
    bypass_vision_with_ground_truth: bool = False
    view_mode: Optional[str] = None
    camera_pitch_deg: Optional[float] = None
    camera_distance: Optional[float] = None
    vision_model: Optional[str] = None
    policy_model: Optional[str] = None


class UiStateRequest(BaseModel):
    """Payload for persisting frontend UI cache for the active game."""

    game_id: str
    updated_at: str
    status_text: str
    has_started: bool
    ai_reason: Optional[str] = None
    last_result: Optional[dict[str, Any]] = None
    event_feed: list[dict[str, Any]] = Field(default_factory=list)
    player_last_move_seconds: float = 0.0
    ai_last_move_seconds: float = 0.0
    player_total_seconds: float = 0.0
    ai_total_seconds: float = 0.0
    vision_model: Optional[str] = None
    policy_model: Optional[str] = None


class _EventHub:
    """Small in-memory pub/sub hub for WebSocket updates."""

    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients = [client for client in self._clients if client != websocket]

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for client in self._clients:
            try:
                await client.send_json(payload)
            except Exception:  # noqa: BLE001
                stale.append(client)
        for client in stale:
            self.disconnect(client)


def _load_yaml_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _file_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    mime, _ = mimetypes.guess_type(str(path))
    content_type = mime or "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _data_url_to_file(data_url: str, out_path: Path) -> Path:
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, payload = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Data URL must be base64 encoded")
    raw = base64.b64decode(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return out_path


def _resolve_image_with_fallback(path: Path) -> Path | None:
    """Resolve an image path with same-basename extension fallbacks."""
    if path.exists():
        return path
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _resolve_base_url(*, configured: str, azure_endpoint_env: str) -> str:
    if configured.strip():
        return configured.strip()
    endpoint = azure_endpoint_env.strip().rstrip("/")
    if not endpoint:
        return ""
    if endpoint.endswith("/openai/v1"):
        return endpoint
    return f"{endpoint}/openai/v1"


def _normalise_piece_placement(raw_value: str) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError("piece placement is empty")
    if " " in value:
        return chess.Board(value).board_fen()
    chess.Board(f"{value} w - - 0 1")
    return value


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value or "").split(",") if item.strip()]


def _first_non_empty_env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _build_azure_model_overrides() -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}

    gpt4o_key = _first_non_empty_env("GPT4o_API_KEY", "GPT4O_API_KEY")
    gpt4o_endpoint = _first_non_empty_env("GPT4o_ENDPOINT", "GPT4O_ENDPOINT")
    gpt4o_api_version = _first_non_empty_env("GPT4o_API_VERSION", "GPT4O_API_VERSION")
    if gpt4o_key or gpt4o_endpoint or gpt4o_api_version:
        overrides["gpt-4o"] = {
            "api_key": gpt4o_key,
            "azure_endpoint": gpt4o_endpoint,
            "api_version": gpt4o_api_version,
        }

    gpt53_key = _first_non_empty_env("GPT53_API_KEY", "GPT5_3_API_KEY")
    gpt53_endpoint = _first_non_empty_env("GPT53_ENDPOINT", "GPT5_3_ENDPOINT")
    gpt53_api_version = _first_non_empty_env("GPT53_API_VERSION", "GPT5_3_API_VERSION")
    if gpt53_key or gpt53_endpoint or gpt53_api_version:
        overrides["gpt-5.3-chat"] = {
            "api_key": gpt53_key,
            "azure_endpoint": gpt53_endpoint,
            "api_version": gpt53_api_version,
        }

    return overrides


def _build_orchestrator_model_options(chess_cfg: dict[str, Any]) -> tuple[str, list[str]]:
    orchestrator_cfg = dict(chess_cfg.get("orchestrator_agent", {}))
    default_model = (
        str(orchestrator_cfg.get("model", "")).strip()
        or os.getenv("AZURE_AGENT_DEPLOYMENT", "").strip()
        or "Qwen/Qwen3-VL-30B-A3B-Instruct"
    )
    model_options = [
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "gpt-5.3-chat",
        "gpt-4o",
        *_split_csv(str(orchestrator_cfg.get("model_options", ""))),
        *_split_csv(str(orchestrator_cfg.get("hf_models", ""))),
    ]
    if default_model:
        model_options.insert(0, default_model)
    seen_models: set[str] = set()
    unique_options = [
        model for model in model_options if not (model in seen_models or seen_models.add(model))
    ]
    if not default_model and unique_options:
        default_model = unique_options[0]
    return default_model, unique_options


def _build_orchestrator_agent_for_model(
    *,
    cfg: dict[str, Any],
    model_name: str,
) -> ChessOrchestratorAgent:
    chess_cfg = dict(cfg.get("chess", {}))
    orchestrator_cfg = dict(chess_cfg.get("orchestrator_agent", {}))
    azure_overrides = _build_azure_model_overrides()

    default_api_key = (
        str(orchestrator_cfg.get("api_key", "")).strip()
        or os.getenv("AZURE_AGENT_API_KEY", "").strip()
    )
    default_base_url = _resolve_base_url(
        configured=str(orchestrator_cfg.get("base_url", "")),
        azure_endpoint_env=os.getenv("AZURE_AGENT_ENDPOINT", "").strip(),
    )
    default_api_version = (
        str(orchestrator_cfg.get("api_version", "")).strip()
        or os.getenv("AZURE_AGENT_API_VERSION", "").strip()
    )
    default_azure_endpoint = (
        str(orchestrator_cfg.get("azure_endpoint", "")).strip()
        or os.getenv("AZURE_AGENT_ENDPOINT", "").strip()
    )

    if model_name.lower().startswith("gpt-"):
        override = azure_overrides.get(model_name, {})
        api_key = str(override.get("api_key", default_api_key)).strip()
        api_version = str(override.get("api_version", default_api_version)).strip()
        azure_endpoint = str(override.get("azure_endpoint", default_azure_endpoint)).strip()
        base_url = str(orchestrator_cfg.get("base_url", "")).strip() or default_base_url
        if not api_key:
            raise ValueError(
                f"Model '{model_name}' requires Azure/OpenAI credentials, but no API key was found."
            )
    else:
        api_key = (
            os.getenv("ORCHESTRATOR_BENCHMARK_API_KEY", "").strip()
            or os.getenv("HUGGING_FACE_API_KEY", "").strip()
        )
        base_url = os.getenv("ORCHESTRATOR_BENCHMARK_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://router.huggingface.co/v1"
        api_version = ""
        azure_endpoint = ""
        if not api_key:
            raise ValueError(
                f"Model '{model_name}' requires Hugging Face credentials, but HUGGING_FACE_API_KEY "
                "or ORCHESTRATOR_BENCHMARK_API_KEY is missing."
            )

    return ChessOrchestratorAgent(
        candidate_count=int(orchestrator_cfg.get("candidate_count", 5)),
        objective_prompt=str(
            orchestrator_cfg.get(
                "objective_prompt",
                "Make the game competitive and instructive while keeping player win chance near target.",
            )
        ),
        model=model_name,
        api_key=api_key,
        base_url=base_url or None,
        api_version=api_version or None,
        azure_endpoint=azure_endpoint or None,
        max_retries=int(orchestrator_cfg.get("max_retries", 2)),
    )


def _build_pipeline(
    cfg: dict[str, Any],
) -> tuple[ChessGameService, ChessMemoryStore, DirectoryCamera, VisionModelResolver]:
    chess_cfg = dict(cfg.get("chess", {}))
    memory_cfg = dict(chess_cfg.get("memory", {}))
    camera_cfg = dict(chess_cfg.get("camera", {}))
    engine_cfg = dict(chess_cfg.get("engine", {}))
    difficulty_cfg = dict(chess_cfg.get("difficulty", {}))
    vision_cfg = dict(chess_cfg.get("vision", {}))
    execution_cfg = dict(chess_cfg.get("execution", {}))

    memory_store = ChessMemoryStore(
        state_path=str(memory_cfg.get("state_path", "data/chess_camera/state/game_state.json")),
        initial_fen=str(
            memory_cfg.get(
                "initial_fen",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            )
        ),
    )
    loaded_state = memory_store.load()
    logger = ChessMoveLogger(
        base_dir=str(cfg.get("run_dir", "games")),
        game_id=str(loaded_state["game_id"]),
    )

    stockfish_service = StockfishService(
        engine_path=str(engine_cfg.get("path", "stockfish")),
        think_time_s=float(engine_cfg.get("think_time_s", 10.0)),
        multipv=int(engine_cfg.get("multipv", 8)),
    )
    difficulty_controller = DifficultyController(
        DifficultyConfig(
            target_player_win_rate=float(difficulty_cfg.get("target_player_win_rate", 0.70)),
            elo_window_moves=int(difficulty_cfg.get("elo_window_moves", 12)),
            elo_prior=int(difficulty_cfg.get("elo_prior", 1000)),
            elo_min=int(difficulty_cfg.get("elo_min", 700)),
            elo_max=int(difficulty_cfg.get("elo_max", 2200)),
            elo_base=int(difficulty_cfg.get("elo_base", 1800)),
            elo_cpl_scale=float(difficulty_cfg.get("elo_cpl_scale", 4.0)),
            elo_confidence_cap=float(difficulty_cfg.get("elo_confidence_cap", 0.90)),
            elo_confidence_power=float(difficulty_cfg.get("elo_confidence_power", 3.0)),
            soft_mode_ratio=float(difficulty_cfg.get("soft_mode_ratio", 0.70)),
            soft_mode_elo_offset=int(difficulty_cfg.get("soft_mode_elo_offset", 120)),
            parity_mode_elo_offset=int(difficulty_cfg.get("parity_mode_elo_offset", 20)),
            max_forced_blunder_cp=int(difficulty_cfg.get("max_forced_blunder_cp", 180)),
            allow_conversion_after_player_blunder_cp=int(
                difficulty_cfg.get("allow_conversion_after_player_blunder_cp", 250)
            ),
            strong_play_avg_cpl_threshold=int(
                difficulty_cfg.get("strong_play_avg_cpl_threshold", 40)
            ),
            strong_play_min_moves=int(difficulty_cfg.get("strong_play_min_moves", 3)),
            close_game_eval_window_cp=int(
                difficulty_cfg.get("close_game_eval_window_cp", 120)
            ),
        )
    )
    orchestrator_default_model, _ = _build_orchestrator_model_options(chess_cfg)
    if not orchestrator_default_model:
        raise ValueError("No default orchestrator model is configured.")
    chess_orchestrator_agent = _build_orchestrator_agent_for_model(
        cfg=cfg,
        model_name=orchestrator_default_model,
    )

    pipeline = ChessGameService(
        memory_store=memory_store,
        logger=logger,
        stockfish_service=stockfish_service,
        chess_orchestrator_agent=chess_orchestrator_agent,
        difficulty_controller=difficulty_controller,
        executor=build_executor(execution_cfg),
        player_colour=str(chess_cfg.get("player_colour", "white")),
        assume_legal_player=bool(chess_cfg.get("assume_legal_player", True)),
    )

    camera = DirectoryCamera(
        inbox_dir=str(camera_cfg.get("inbox_dir", "data/chess_camera/inbox")),
        current_filename=str(camera_cfg.get("current_filename", "camera_capture.jpg")),
    )

    vision_model = (
        str(vision_cfg.get("model", "")).strip()
        or os.getenv("AZURE_VISION_DEPLOYMENT", "").strip()
        or "gpt-5.3-chat"
    )
    vision_api_key = (
        str(vision_cfg.get("api_key", "")).strip()
        or os.getenv("AZURE_VISION_API_KEY", "").strip()
    )
    vision_base_url = _resolve_base_url(
        configured=str(vision_cfg.get("base_url", "")),
        azure_endpoint_env=os.getenv("AZURE_VISION_ENDPOINT", "").strip(),
    )
    vision_api_version = (
        str(vision_cfg.get("api_version", "")).strip()
        or os.getenv("AZURE_VISION_API_VERSION", "").strip()
    )
    vision_azure_endpoint = (
        str(vision_cfg.get("azure_endpoint", "")).strip()
        or os.getenv("AZURE_VISION_ENDPOINT", "").strip()
    )
    max_retries = int(vision_cfg.get("max_retries", 2))

    azure_allowlist_raw = str(vision_cfg.get("azure_model_allowlist", "")).strip()
    azure_model_allowlist = {
        item.strip() for item in azure_allowlist_raw.split(",") if item.strip()
    }
    if vision_model:
        azure_model_allowlist.add(vision_model)

    model_resolver = VisionModelResolver(
        default_model=vision_model,
        settings=VisionProviderSettings(
            azure_api_key=vision_api_key,
            azure_base_url=vision_base_url,
            azure_api_version=vision_api_version,
            azure_endpoint=vision_azure_endpoint,
            azure_max_retries=max_retries,
            azure_timeout_s=float(vision_cfg.get("azure_timeout_s", 120.0)),
            hugging_face_api_key=os.getenv("HUGGING_FACE_API_KEY", "").strip(),
            hugging_face_max_retries=int(vision_cfg.get("hf_max_retries", 4)),
            hugging_face_timeout_s=float(vision_cfg.get("hf_timeout_s", 90.0)),
            azure_model_allowlist=azure_model_allowlist,
            azure_model_overrides=_build_azure_model_overrides(),
        ),
    )

    return pipeline, memory_store, camera, model_resolver


def create_app(config_path: str = "configs/chess_move.yaml") -> FastAPI:
    """Create and configure the orchestrator API application."""

    cfg = _load_yaml_config(config_path)
    chess_cfg = dict(cfg.get("chess", {}))
    camera_cfg = dict(chess_cfg.get("camera", {}))
    camera_input_mode = str(camera_cfg.get("input_mode", "filesystem")).strip().lower() or "filesystem"
    if camera_input_mode == "frontend_ui":
        camera_input_mode = "ui_render"
    if camera_input_mode not in {"filesystem", "ui_render"}:
        camera_input_mode = "filesystem"
    vision_cfg = dict(chess_cfg.get("vision", {}))
    vision_default_model = (
        str(vision_cfg.get("model", "")).strip()
        or os.getenv("AZURE_VISION_DEPLOYMENT", "").strip()
        or "gpt-5.3-chat"
    )
    vision_model_options = [
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "gpt-5.3-chat",
        "gpt-4o",
        *_split_csv(str(vision_cfg.get("azure_model_allowlist", ""))),
        *_split_csv(str(vision_cfg.get("hf_models", ""))),
    ]
    if vision_default_model:
        vision_model_options.insert(0, vision_default_model)
    # Preserve order while deduplicating.
    seen_models: set[str] = set()
    vision_model_options = [
        model for model in vision_model_options if not (model in seen_models or seen_models.add(model))
    ]
    legal_retry_attempts = max(1, int(vision_cfg.get("illegal_retry_attempts", 3)))
    pipeline, memory_store, camera, model_resolver = _build_pipeline(cfg)
    policy_default_model, policy_model_options = _build_orchestrator_model_options(chess_cfg)
    policy_agent_cache: dict[str, ChessOrchestratorAgent] = {}
    if policy_default_model:
        policy_agent_cache[policy_default_model] = pipeline.chess_orchestrator_agent

    def _resolve_policy_agent(
        requested_model: str | None,
    ) -> tuple[str, ChessOrchestratorAgent]:
        selected_model = str(requested_model or "").strip() or policy_default_model
        if not selected_model:
            raise ValueError("No policy model was requested and no default policy model is configured.")
        if selected_model in policy_agent_cache:
            return selected_model, policy_agent_cache[selected_model]
        agent = _build_orchestrator_agent_for_model(cfg=cfg, model_name=selected_model)
        policy_agent_cache[selected_model] = agent
        return selected_model, agent

    events = _EventHub()

    app = FastAPI(title="LLM VLA Chess Orchestrator", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/state")
    async def get_state() -> dict[str, Any]:
        state = memory_store.load()
        pipeline.logger.ensure_game(str(state["game_id"]))
        return {
            "game_id": state["game_id"],
            "current_fen": state["current_fen"],
            "initial_fen": state["initial_fen"],
            "move_index": state["move_index"],
            "pending_illegal_transition": state.get("pending_illegal_transition"),
            "stats": state["memory"]["stats"],
            "player_history_len": len(state.get("player_history", [])),
            "camera_input_mode": camera_input_mode,
            "vision_default_model": vision_default_model,
            "vision_model_options": vision_model_options,
            "policy_default_model": policy_default_model,
            "policy_model_options": policy_model_options,
            "ui_state": pipeline.logger.load_ui_state(),
        }

    @app.post("/api/reset")
    async def reset_game() -> dict[str, Any]:
        state = memory_store.reset(reason="api_reset")
        pipeline.logger.ensure_game(str(state["game_id"]))
        pipeline.logger.save_ui_state({})
        payload = {
            "status": "reset",
            "game_id": state["game_id"],
            "current_fen": state["current_fen"],
            "move_index": state["move_index"],
        }
        await events.broadcast({"event": "game_reset", "data": payload})
        return payload

    @app.post("/api/ui/state")
    async def save_ui_state(request: UiStateRequest) -> dict[str, Any]:
        state = memory_store.load()
        game_id = str(state["game_id"])
        if str(request.game_id) != game_id:
            raise HTTPException(
                status_code=409,
                detail=f"UI cache game_id mismatch. Expected {game_id}, got {request.game_id}.",
            )
        pipeline.logger.ensure_game(game_id)
        pipeline.logger.save_ui_state(
            {
                "game_id": game_id,
                "updated_at": request.updated_at,
                "status_text": request.status_text,
                "has_started": request.has_started,
                "ai_reason": str(request.ai_reason or "").strip() or None,
                "last_result": request.last_result,
                "event_feed": request.event_feed[:20],
                "player_last_move_seconds": float(request.player_last_move_seconds),
                "ai_last_move_seconds": float(request.ai_last_move_seconds),
                "player_total_seconds": float(request.player_total_seconds),
                "ai_total_seconds": float(request.ai_total_seconds),
                "vision_model": str(request.vision_model or "").strip() or None,
                "policy_model": str(request.policy_model or "").strip() or None,
            }
        )
        return {"status": "ok", "game_id": game_id}

    @app.post("/api/player/analyse")
    @traceable(name="chess_pipeline", run_type="chain")
    async def move(request: AnalyseRequest) -> dict[str, Any]:
        LOGGER.info(
            "api/player/analyse: received request camera_input_mode=%s",
            camera_input_mode,
        )
        observed_piece_placement: str | None = None
        vision_attempts_used = 0
        analysis_image_data_url = request.analysis_image_data_url
        predicted_history: list[dict[str, Any]] = []
        vision_model_used: str | None = None
        policy_model_used: str | None = None

        if camera_input_mode == "ui_render":
            ui_image_path = Path(camera.inbox_dir) / "frontend_capture.jpg"
            if request.analysis_image_data_url:
                try:
                    image_path = _data_url_to_file(request.analysis_image_data_url, ui_image_path)
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(status_code=400, detail=f"Invalid frontend snapshot: {exc}") from exc
                analysis_image_data_url = request.analysis_image_data_url
            else:
                resolved = _resolve_image_with_fallback(ui_image_path)
                if resolved is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "camera.input_mode is ui_render, but no analysis image was provided and "
                            "no saved frontend_capture file (.jpg/.png/...) exists."
                        ),
                    )
                image_path = resolved
            LOGGER.info(
                "api/player/analyse: using ui_render snapshot at %s",
                image_path,
            )
        elif request.image_path:
            image_path = Path(request.image_path)
        else:
            image_path = camera.get_current_image()
        if camera_input_mode == "filesystem":
            LOGGER.info(
                "api/player/analyse: using filesystem image at %s",
                image_path,
            )
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image does not exist: {image_path}")
        if analysis_image_data_url is None:
            analysis_image_data_url = _file_to_data_url(image_path)
        state = memory_store.load()
        before_fen = str(state["current_fen"])
        board_before = chess.Board(before_fen)
        ground_truth_piece_placement: str | None = None
        ground_truth_move_san: str | None = None

        if request.ground_truth_piece_placement:
            try:
                ground_truth_piece_placement = _normalise_piece_placement(
                    request.ground_truth_piece_placement
                )
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Invalid ground-truth piece placement: {exc}") from exc
            ground_truth_validation = ChessGameService._validate_transition(
                board_before=board_before,
                observed_piece_placement=ground_truth_piece_placement,
            )
            if not ground_truth_validation.is_legal:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provided ground-truth piece placement is not a legal single-move transition "
                        "from current state."
                    ),
                )
            ground_truth_move_san = str(ground_truth_validation.matched_move_san)

        if request.bypass_vision_with_ground_truth:
            if not ground_truth_piece_placement:
                raise HTTPException(
                    status_code=400,
                    detail="bypass_vision_with_ground_truth requires ground_truth_piece_placement.",
                )
            observed_piece_placement = ground_truth_piece_placement
            LOGGER.info("api/player/analyse: bypassing vision and using provided ground truth")
        else:
            feedback: str | None = None
            max_legal_attempts = legal_retry_attempts
            latest_recognition = None
            requested_vision_model = str(request.vision_model or "").strip() or None
            try:
                resolved_model_name, recogniser = model_resolver.resolve(requested_vision_model)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            vision_model_used = resolved_model_name

            for _ in range(max_legal_attempts):
                LOGGER.info("api/player/analyse: running vision recognition attempt")
                recognition, attempts_used = recogniser.recognise_move(
                    image_path=str(image_path),
                    before_fen=before_fen,
                    feedback=feedback,
                )
                vision_attempts_used += attempts_used
                latest_recognition = recognition

                placement_validation = ChessGameService._validate_transition(
                    board_before=board_before,
                    observed_piece_placement=recognition.after_piece_placement,
                )

                recognised_move = None
                try:
                    recognised_move = board_before.parse_san(recognition.move_san)
                except ValueError:
                    recognised_move = None

                inferred_san_from_placement = placement_validation.matched_move_san
                predicted_history.append(
                    {
                        "attempt": len(predicted_history) + 1,
                        "vision_model": resolved_model_name,
                        "predicted_move_san": recognition.move_san,
                        "predicted_after_piece_placement": recognition.after_piece_placement,
                        "san_is_legal": recognised_move is not None,
                        "after_piece_placement_is_legal": placement_validation.is_legal,
                        "inferred_move_san_from_placement": inferred_san_from_placement,
                    }
                )

                if recognised_move is not None and placement_validation.is_legal:
                    san_uci = recognised_move.uci()
                    placement_uci = str(placement_validation.matched_move_uci)
                    if san_uci == placement_uci:
                        observed_piece_placement = recognition.after_piece_placement
                        LOGGER.info("api/player/analyse: vision SAN and after placement agree on legal move")
                        break

                if placement_validation.is_legal:
                    observed_piece_placement = recognition.after_piece_placement
                    LOGGER.info("api/player/analyse: using legal after_piece_placement from vision output")
                    break

                if recognised_move is not None:
                    trial_board = board_before.copy(stack=False)
                    trial_board.push(recognised_move)
                    observed_piece_placement = trial_board.board_fen()
                    LOGGER.info("api/player/analyse: using legal SAN move from vision output")
                    break

                LOGGER.warning("api/player/analyse: vision output was illegal, requesting retry")
                previous_predictions = ", ".join(
                    str(item.get("predicted_move_san") or "?") for item in predicted_history
                )
                feedback = (
                    "This is not what the player moved. "
                    f"Previous predicted SAN moves: [{previous_predictions}]. "
                    f"Your latest SAN '{recognition.move_san}' is not legal from before_fen '{before_fen}', "
                    f"and after_piece_placement '{recognition.after_piece_placement}' is not a legal "
                    "single-move transition. "
                    "Try again and output one legal move with matching SAN and placement."
                )

            if latest_recognition is None:
                raise HTTPException(status_code=500, detail="Vision recogniser returned no result.")
            if not observed_piece_placement:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Vision could not identify a legal SAN move from the provided before_fen "
                        "and after-move image."
                    ),
                )

        if ground_truth_piece_placement and observed_piece_placement != ground_truth_piece_placement:
            last_prediction = predicted_history[-1] if predicted_history else {}
            predicted_move_san = str(last_prediction.get("predicted_move_san", "")).strip() or None
            message = (
                f"Vision prediction did not match the player's move. "
                f"Predicted move: {predicted_move_san or 'unknown'}"
            )
            return {
                "status": "vision_mismatch_error",
                "can_bypass": True,
                "message": message,
                "predicted_move_san": predicted_move_san,
                "predicted_piece_placement": observed_piece_placement,
                "ground_truth_move_san": ground_truth_move_san,
                "ground_truth_piece_placement": ground_truth_piece_placement,
                "predictions": predicted_history,
                "vision_attempts_used": vision_attempts_used,
                "vision_model": vision_model_used,
            }

        if vision_attempts_used == 0:
            vision_attempts_used = 1

        LOGGER.info("api/player/analyse: invoking chess game service move processing")
        requested_policy_model = str(request.policy_model or "").strip() or None
        try:
            policy_model_used, policy_agent = _resolve_policy_agent(requested_policy_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result = pipeline.move(
            observed_piece_placement=observed_piece_placement,
            player_time_s=request.player_time_s,
            policy_agent_override=policy_agent,
            override_illegal=False,
            source="camera",
            vision_attempts_used=vision_attempts_used,
            analysis_image_data_url=analysis_image_data_url,
            view_mode=request.view_mode,
            camera_pitch_deg=request.camera_pitch_deg,
            camera_distance=request.camera_distance,
        )
        result["vision_model"] = vision_model_used
        result["policy_model"] = policy_model_used
        LOGGER.info("api/player/analyse: completed status=%s move_index=%s", result.get("status"), result.get("move_index"))
        await events.broadcast({"event": "move_analysed", "data": result})
        return result

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        await events.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            events.disconnect(websocket)

    return app
