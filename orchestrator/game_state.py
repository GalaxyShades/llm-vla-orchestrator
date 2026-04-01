"""Game state storage for chess move orchestration."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")


class ChessMemoryStore:
    """Persists game state, policy evidence, and rich event history across moves."""

    def __init__(self, state_path: str, initial_fen: str) -> None:
        self.state_path = Path(state_path)
        self.initial_fen = initial_fen

    def _new_state(self) -> dict[str, Any]:
        game_id = self._new_game_id()
        return {
            "schema_version": 3,
            "game_id": game_id,
            "current_fen": self.initial_fen,
            "initial_fen": self.initial_fen,
            "move_index": 0,
            "moves_uci": [],
            "pending_illegal_transition": None,
            "player_history": [],
            "memory": {
                "journal": [],
                "events": [],
                "stats": {
                    "total_moves": 0,
                    "legal_moves": 0,
                    "illegal_moves": 0,
                    "overrides": 0,
                },
                "metadata": {},
            },
        }

    def _normalise_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state.setdefault("schema_version", 3)
        state.setdefault("game_id", self._new_game_id())
        state.setdefault("current_fen", self.initial_fen)
        state.setdefault("initial_fen", self.initial_fen)
        state.setdefault("move_index", 0)
        state.setdefault("moves_uci", [])
        state.setdefault("pending_illegal_transition", None)
        state.setdefault("player_history", [])

        memory = state.setdefault("memory", {})
        memory.setdefault("journal", [])
        memory.setdefault("events", [])
        stats = memory.setdefault("stats", {})
        stats.setdefault("total_moves", 0)
        stats.setdefault("legal_moves", 0)
        stats.setdefault("illegal_moves", 0)
        stats.setdefault("overrides", 0)
        memory.setdefault("metadata", {})
        return state

    def load(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            state = self._normalise_state(state)
            self.save(state)
            return state

        state = self._new_state()
        self.save(state)
        return state

    def reset(self, reason: str = "manual") -> dict[str, Any]:
        state = self._new_state()
        state["memory"]["metadata"]["last_reset_reason"] = reason
        state["memory"]["metadata"]["last_reset_at"] = datetime.now(HONG_KONG_TZ).isoformat()
        self.save(state)
        return state

    def save(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)

    @staticmethod
    def _new_game_id() -> str:
        return datetime.now(HONG_KONG_TZ).strftime("%Y%m%d_%H%M%S")
