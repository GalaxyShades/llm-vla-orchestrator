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
from orchestrator.executor import PiZeroExecutor
from orchestrator.game_logger import ChessMoveLogger
from orchestrator.game_service import ChessGameService
from orchestrator.game_state import ChessMemoryStore
from orchestrator.policy_agent import ChessOrchestratorAgent
from orchestrator.vision_agent import ChatGPTVisionRecognizer

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


def _build_pipeline(
    cfg: dict[str, Any],
) -> tuple[ChessGameService, ChessMemoryStore, DirectoryCamera, Optional[ChatGPTVisionRecognizer]]:
    chess_cfg = dict(cfg.get("chess", {}))
    memory_cfg = dict(chess_cfg.get("memory", {}))
    camera_cfg = dict(chess_cfg.get("camera", {}))
    engine_cfg = dict(chess_cfg.get("engine", {}))
    difficulty_cfg = dict(chess_cfg.get("difficulty", {}))
    vision_cfg = dict(chess_cfg.get("vision", {}))

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
    orchestrator_agent_cfg = dict(chess_cfg.get("orchestrator_agent", {}))
    orchestrator_model = (
        str(orchestrator_agent_cfg.get("model", "")).strip()
        or os.getenv("AZURE_AGENT_DEPLOYMENT", "").strip()
    )
    orchestrator_api_key = (
        str(orchestrator_agent_cfg.get("api_key", "")).strip()
        or os.getenv("AZURE_AGENT_API_KEY", "").strip()
    )
    orchestrator_base_url = _resolve_base_url(
        configured=str(orchestrator_agent_cfg.get("base_url", "")),
        azure_endpoint_env=os.getenv("AZURE_AGENT_ENDPOINT", "").strip(),
    )
    orchestrator_api_version = (
        str(orchestrator_agent_cfg.get("api_version", "")).strip()
        or os.getenv("AZURE_AGENT_API_VERSION", "").strip()
    )
    orchestrator_azure_endpoint = (
        str(orchestrator_agent_cfg.get("azure_endpoint", "")).strip()
        or os.getenv("AZURE_AGENT_ENDPOINT", "").strip()
    )
    chess_orchestrator_agent = ChessOrchestratorAgent(
        candidate_count=int(orchestrator_agent_cfg.get("candidate_count", 5)),
        objective_prompt=str(
            orchestrator_agent_cfg.get(
                "objective_prompt",
                "Make the game competitive and instructive while keeping player win chance near target.",
            )
        ),
        model=orchestrator_model,
        api_key=orchestrator_api_key,
        base_url=orchestrator_base_url or None,
        api_version=orchestrator_api_version or None,
        azure_endpoint=orchestrator_azure_endpoint or None,
        max_retries=int(orchestrator_agent_cfg.get("max_retries", 2)),
    )

    pipeline = ChessGameService(
        memory_store=memory_store,
        logger=logger,
        stockfish_service=stockfish_service,
        chess_orchestrator_agent=chess_orchestrator_agent,
        difficulty_controller=difficulty_controller,
        executor=PiZeroExecutor(),
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

    recogniser: Optional[ChatGPTVisionRecognizer] = None
    if vision_model and vision_api_key:
        recogniser = ChatGPTVisionRecognizer(
            model=vision_model,
            api_key=vision_api_key,
            base_url=vision_base_url or None,
            api_version=vision_api_version or None,
            azure_endpoint=vision_azure_endpoint or None,
            max_retries=max_retries,
        )

    return pipeline, memory_store, camera, recogniser


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
    legal_retry_attempts = max(1, int(vision_cfg.get("illegal_retry_attempts", 3)))
    pipeline, memory_store, camera, recogniser = _build_pipeline(cfg)
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

        if recogniser is None:
            raise HTTPException(
                status_code=400,
                detail="Vision recogniser is not configured. Set chess.vision.model and API key.",
            )
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
                    + (
                        f"The player move SAN was '{ground_truth_move_san}'. "
                        if ground_truth_move_san
                        else ""
                    )
                    + "Try again and output one legal move with matching SAN and placement."
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
            }

        if vision_attempts_used == 0:
            vision_attempts_used = 1

        LOGGER.info("api/player/analyse: invoking chess game service move processing")
        result = pipeline.move(
            observed_piece_placement=observed_piece_placement,
            player_time_s=request.player_time_s,
            override_illegal=False,
            source="camera",
            vision_attempts_used=vision_attempts_used,
            analysis_image_data_url=analysis_image_data_url,
            view_mode=request.view_mode,
            camera_pitch_deg=request.camera_pitch_deg,
            camera_distance=request.camera_distance,
        )
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
