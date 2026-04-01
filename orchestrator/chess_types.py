"""Shared data structures for chess orchestration services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VisionMoveState:
    """Normalised vision move prediction returned by analysis."""

    after_piece_placement: str
    move_san: str
    overall_confidence: float | None
    raw_model_output: str


@dataclass
class EngineCandidate:
    """Single candidate move from Stockfish analysis."""

    uci: str
    san: str
    eval_cp: int
    cp_loss: int


@dataclass
class EngineMoveSelection:
    """Engine choice and candidate set for policy selection."""

    best_eval_cp: int
    selected: EngineCandidate
    candidates: list[EngineCandidate]


@dataclass
class ChessOrchestratorDecision:
    """Decision payload returned by the chess orchestrator agent."""

    selected: EngineCandidate
    reason: str
    candidate_scores: dict[str, float]


@dataclass
class DifficultyConfig:
    """Tunable policy controls for adaptive play strength."""

    target_player_win_rate: float = 0.70
    elo_window_moves: int = 12
    elo_prior: int = 1000
    elo_min: int = 700
    elo_max: int = 2200
    elo_base: int = 1800
    elo_cpl_scale: float = 4.0
    elo_confidence_cap: float = 0.90
    soft_mode_ratio: float = 0.70
    soft_mode_elo_offset: int = 120
    parity_mode_elo_offset: int = 20
    max_forced_blunder_cp: int = 180
    allow_conversion_after_player_blunder_cp: int = 250
    strong_play_avg_cpl_threshold: int = 40
    strong_play_min_moves: int = 3
    close_game_eval_window_cp: int = 120


@dataclass
class PlayerMoveEvidence:
    """Data points describing the player's latest move quality."""

    move_uci: str
    move_san: str
    eval_before_cp: int
    eval_after_cp: int
    centipawn_loss: int
    player_time_s: float | None


@dataclass
class TransitionValidation:
    """Result of validating an observed board against legal moves."""

    is_legal: bool
    matched_move_uci: str | None
    matched_move_san: str | None
    warning: str | None


@dataclass
class ChessMoveRecord:
    """Detailed per-move record persisted as JSONL."""

    timestamp: str
    move_index: int
    status: str
    pre_fen: str
    observed_piece_placement: str
    player_move_uci: str | None
    ai_move_uci: str | None
    ai_move_san: str | None
    post_fen: str
    warning: str | None
    override_used: bool
    vision_attempts_used: int
    player_move_evidence: dict[str, Any] | None
    player_estimated_elo: int
    policy_mode: str
    policy_context: dict[str, Any]
    stockfish_context: dict[str, Any]
    pi_instruction: str | None
    execution_verified: bool
    analysis_image_path: str | None
    logs: dict[str, Any]
