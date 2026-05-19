"""Tests for the refactored chess orchestration pipeline with legality-first control."""

from __future__ import annotations

import json
from pathlib import Path

import chess

from orchestrator.chess_types import (
    ChessOrchestratorDecision,
    DifficultyConfig,
    EngineCandidate,
    EngineMoveSelection,
    PlayerMoveEvidence,
)
from orchestrator.difficulty import DifficultyController
from orchestrator.executor import PiZeroExecutor
from orchestrator.executor import build_executor
from orchestrator.game_service import ChessGameService
from orchestrator.game_logger import ChessMoveLogger
from orchestrator.game_state import ChessMemoryStore


class FakeStockfishService:
    """Deterministic fake engine for fast and reproducible unit tests."""

    multipv = 8

    def analyse_move_quality(self, board_before: chess.Board, move: chess.Move) -> PlayerMoveEvidence:
        return PlayerMoveEvidence(
            move_uci=move.uci(),
            move_san=board_before.san(move),
            eval_before_cp=40,
            eval_after_cp=5,
            centipawn_loss=35,
            player_time_s=None,
        )

    def get_top_move_candidates(self, *, board: chess.Board, top_k: int = 5) -> EngineMoveSelection:
        del top_k
        move = next(iter(board.legal_moves))
        candidate = EngineCandidate(
            uci=move.uci(),
            san=board.san(move),
            eval_cp=12,
            cp_loss=10,
        )
        return EngineMoveSelection(
            best_eval_cp=22,
            selected=candidate,
            candidates=[candidate],
        )


class FakePolicyAgent:
    """Deterministic policy agent for test runs without network calls."""

    candidate_count = 5

    def choose_move(
        self,
        *,
        candidates: list[EngineCandidate],
        best_eval_cp: int,
        player_estimated_elo: int,
        policy_mode: str,
        game_objective: str,
        close_game_eval_window_cp: int,
        target_cp_loss: int,
        target_player_win_rate: float,
        allow_best_play: bool,
        player_move_evidence: dict[str, float] | None,
        stats: dict[str, int] | None = None,
    ) -> ChessOrchestratorDecision:
        del best_eval_cp
        del player_estimated_elo
        del policy_mode
        del game_objective
        del close_game_eval_window_cp
        del target_cp_loss
        del target_player_win_rate
        del allow_best_play
        del player_move_evidence
        del stats

        selected = candidates[0]
        return ChessOrchestratorDecision(
            selected=selected,
            reason="Selected first candidate for deterministic test.",
            candidate_scores={selected.uci: 1.0},
        )


def _build_pipeline(*, tmp_path: Path, initial_fen: str = chess.STARTING_FEN) -> ChessGameService:
    memory_store = ChessMemoryStore(
        state_path=str(tmp_path / "state" / "game_state.json"),
        initial_fen=initial_fen,
    )
    logger = ChessMoveLogger(base_dir=str(tmp_path / "games"))
    difficulty_controller = DifficultyController(DifficultyConfig(elo_window_moves=12))
    return ChessGameService(
        memory_store=memory_store,
        logger=logger,
        stockfish_service=FakeStockfishService(),
        chess_orchestrator_agent=FakePolicyAgent(),
        difficulty_controller=difficulty_controller,
        executor=PiZeroExecutor(),
        player_colour="white",
    )


def test_executor_factory_defaults_to_dry_run() -> None:
    executor = build_executor({})

    assert isinstance(executor, PiZeroExecutor)
    assert executor.execute_move("e2e4") == (True, "Move piece from e2 to e4")


def test_pipeline_accepts_legal_transition_and_updates_memory(tmp_path: Path) -> None:
    board = chess.Board()
    board.push_san("e4")
    observed_piece_placement = board.board_fen()
    pipeline = _build_pipeline(tmp_path=tmp_path)

    result = pipeline.move(
        observed_piece_placement=observed_piece_placement,
        player_time_s=4.2,
        source="simulated",
        vision_attempts_used=1,
    )

    assert result["status"] == "ok"
    assert result["player_move_uci"] == "e2e4"
    assert result["ai_move_uci"] is not None

    state_path = tmp_path / "state" / "game_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["move_index"] == 1
    assert len(state["moves_uci"]) == 2

    boards_dir = Path(result["boards_dir"])
    assert (boards_dir / "move_001_pre.png").exists()
    assert (boards_dir / "move_001_observed.png").exists()
    assert (boards_dir / "move_001_post.png").exists()


def test_pipeline_infers_most_likely_legal_transition_when_observation_is_illegal(tmp_path: Path) -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    illegal_piece_placement = board.board_fen()

    pipeline = _build_pipeline(tmp_path=tmp_path)

    result = pipeline.move(
        observed_piece_placement=illegal_piece_placement,
        player_time_s=5.5,
        source="simulated",
        vision_attempts_used=1,
    )

    assert result["status"] == "ok"
    assert result["player_move_uci"] == "e2e4"
    assert "most likely legal move was inferred" in str(result["warning"])

    state_path = tmp_path / "state" / "game_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["pending_illegal_transition"] is None
    assert state["memory"]["stats"]["overrides"] == 0
