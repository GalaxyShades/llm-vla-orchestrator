"""Difficulty control and player-strength estimation for chess orchestration."""

from __future__ import annotations

import random
from typing import Any

from langsmith import traceable

from orchestrator.chess_types import DifficultyConfig


class DifficultyController:
    """Computes policy mode and target strength from recent player evidence."""

    def __init__(self, config: DifficultyConfig) -> None:
        self.config = config

    @traceable(name="chess_difficulty_estimate_player_elo", run_type="tool")
    def estimate_player_elo(self, player_history: list[dict[str, Any]]) -> int:
        prior = int(self.config.elo_prior)
        if not player_history:
            return prior

        recent = player_history[-self.config.elo_window_moves :]
        cpl_values = [max(0, int(item.get("centipawn_loss", 120))) for item in recent]
        weights = list(range(1, len(cpl_values) + 1))
        weighted_cpl = sum(cpl * w for cpl, w in zip(cpl_values, weights)) / sum(weights)

        model_estimate = int(self.config.elo_base - (weighted_cpl * float(self.config.elo_cpl_scale)))
        model_estimate = max(int(self.config.elo_min), min(int(self.config.elo_max), model_estimate))

        confidence = min(
            float(self.config.elo_confidence_cap),
            (len(recent) / max(1, int(self.config.elo_window_moves)))
            ** float(self.config.elo_confidence_power),
        )
        blended = int(prior + confidence * (model_estimate - prior))
        return max(int(self.config.elo_min), min(int(self.config.elo_max), blended))

    @traceable(name="chess_difficulty_sample_game_objective", run_type="tool")
    def sample_game_objective(self) -> str:
        player_target = max(0.0, min(1.0, float(self.config.target_player_win_rate)))
        return "ai_should_lose" if random.random() < player_target else "ai_should_win"

    @traceable(name="chess_difficulty_choose_policy_mode", run_type="tool")
    def choose_policy_mode(self, game_objective: str) -> str:
        if game_objective == "ai_should_lose":
            return "soft_mode"
        return "parity_mode"

    @traceable(name="chess_difficulty_resolve_effective_objective", run_type="tool")
    def resolve_effective_objective(
        self,
        *,
        game_objective: str,
        player_history: list[dict[str, Any]],
        latest_player_move_evidence: dict[str, Any] | None,
    ) -> tuple[str, str]:
        effective_objective = game_objective
        reason = "base_objective"

        if (
            game_objective == "ai_should_lose"
            and isinstance(latest_player_move_evidence, dict)
            and int(latest_player_move_evidence.get("centipawn_loss", 0))
            >= self.config.allow_conversion_after_player_blunder_cp
        ):
            return "ai_should_win", "override_grave_blunder"

        if game_objective == "ai_should_win":
            recent = player_history[-self.config.elo_window_moves :]
            cpl_values = [int(item.get("centipawn_loss", 120)) for item in recent]
            if len(cpl_values) >= self.config.strong_play_min_moves:
                avg_cpl = sum(cpl_values) / len(cpl_values)
                if avg_cpl <= self.config.strong_play_avg_cpl_threshold:
                    return "ai_should_lose", "override_exceptional_play"

        return effective_objective, reason

    @traceable(name="chess_difficulty_target_cp_loss", run_type="tool")
    def target_cp_loss(self, policy_mode: str) -> int:
        if policy_mode == "soft_mode":
            return min(
                self.config.max_forced_blunder_cp,
                max(15, int(self.config.soft_mode_elo_offset * 1.1)),
            )
        return min(
            self.config.max_forced_blunder_cp,
            max(5, int(self.config.parity_mode_elo_offset * 0.45)),
        )
