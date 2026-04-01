"""Entrypoint for running chess move orchestration pipelines."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from orchestrator.game_service import ChessGameService
from orchestrator.chess_types import DifficultyConfig
from orchestrator.difficulty import DifficultyController
from orchestrator.engine_service import StockfishService
from orchestrator.executor import PiZeroExecutor
from orchestrator.game_logger import ChessMoveLogger
from orchestrator.game_state import ChessMemoryStore
from orchestrator.policy_agent import ChessOrchestratorAgent
from orchestrator.web_app import create_app


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def apply_langsmith_settings(cfg: dict[str, Any]) -> None:
    ls_cfg = cfg.get("langsmith", {})
    if not isinstance(ls_cfg, dict):
        return

    enabled = ls_cfg.get("enabled")
    if enabled is not None:
        os.environ["LANGSMITH_TRACING"] = "true" if bool(enabled) else "false"
    else:
        os.environ.setdefault("LANGSMITH_TRACING", "true")

    project = ls_cfg.get("project")
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", str(project))

    api_key = ls_cfg.get("api_key")
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", str(api_key))

    endpoint = ls_cfg.get("endpoint")
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", str(endpoint))

    # Preserve explicit env overrides when already set by caller.


def _resolve_base_url(*, configured: str, azure_endpoint_env: str) -> str:
    if configured.strip():
        return configured.strip()
    endpoint = azure_endpoint_env.strip().rstrip("/")
    if not endpoint:
        return ""
    if endpoint.endswith("/openai/v1"):
        return endpoint
    return f"{endpoint}/openai/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task orchestration pipeline")
    parser.add_argument(
        "--config",
        default="configs/chess_move.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--reset-game-state",
        action="store_true",
        help="Reset chess game state before running chess_move pipeline.",
    )
    parser.add_argument(
        "--move-note",
        default="",
        help="Optional note attached to this chess move and stored in memory.",
    )
    parser.add_argument(
        "--observed-piece-placement",
        default="",
        help="Board-FEN piece placement after the player's move (simulated mode).",
    )
    parser.add_argument(
        "--player-time-s",
        type=float,
        default=None,
        help="Player move time in seconds for this move.",
    )
    parser.add_argument(
        "--override-illegal",
        action="store_true",
        help="Override illegal transition check for this move.",
    )
    parser.add_argument(
        "--serve-api",
        action="store_true",
        help="Run the FastAPI chess service instead of one-shot pipeline execution.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --serve-api mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve-api mode.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    load_env_file(".env")
    cfg = load_config(args.config)
    apply_langsmith_settings(cfg)

    if args.serve_api:
        import uvicorn

        app = create_app(config_path=args.config)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    _run_chess_move_pipeline(
        cfg,
        reset_game_state=bool(args.reset_game_state),
        move_note=str(args.move_note or "").strip() or None,
        observed_piece_placement=str(args.observed_piece_placement or "").strip() or None,
        player_time_s=args.player_time_s,
        override_illegal=bool(args.override_illegal),
    )


def _run_chess_move_pipeline(
    cfg: dict[str, Any],
    *,
    reset_game_state: bool = False,
    move_note: str | None = None,
    observed_piece_placement: str | None = None,
    player_time_s: float | None = None,
    override_illegal: bool = False,
) -> None:
    run_dir = str(cfg.get("run_dir", "games"))
    chess_cfg = cfg.get("chess", {})
    memory_cfg = chess_cfg.get("memory", {})
    difficulty_cfg = chess_cfg.get("difficulty", {})
    engine_cfg = chess_cfg.get("engine", {})

    memory_store = ChessMemoryStore(
        state_path=str(memory_cfg.get("state_path", "data/chess_camera/state/game_state.json")),
        initial_fen=str(
            memory_cfg.get(
                "initial_fen",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            )
        ),
    )
    if bool(memory_cfg.get("reset_on_start", False)) or reset_game_state:
        reset_reason = "cli_flag" if reset_game_state else "config_reset_on_start"
        memory_store.reset(reason=reset_reason)
    logger = ChessMoveLogger(base_dir=run_dir)
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
            close_game_eval_window_cp=int(difficulty_cfg.get("close_game_eval_window_cp", 120)),
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
    )

    if not observed_piece_placement:
        raise ValueError(
            "--observed-piece-placement is required for one-shot chess_move execution. "
            "Use --serve-api for full UI mode."
        )
    del move_note
    result = pipeline.move(
        observed_piece_placement=observed_piece_placement,
        player_time_s=player_time_s,
        override_illegal=override_illegal,
        source="simulated",
        vision_attempts_used=1,
    )

    print("Pipeline: chess_move")
    print(f"Status: {result['status']}")
    print(f"Move index: {result['move_index']}")
    print(f"Run directory: {result['run_dir']}")
    print(f"Moves log: {result['moves_log']}")
    print(f"PGN path: {result['pgn_path']}")
    if result["player_move_uci"]:
        print(f"Detected player move (UCI): {result['player_move_uci']}")
    else:
        print("Detected player move (UCI): none")
    print(f"Selected AI move (UCI): {result['ai_move_uci']}")


if __name__ == "__main__":
    main()
