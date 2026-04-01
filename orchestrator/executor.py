"""Execution adapters for applying chess moves on external hardware."""

from __future__ import annotations

from langsmith import traceable


class PiZeroExecutor:
    """Stub executor for Pi Zero move commands in development."""

    @traceable(name="chess_executor_execute_move", run_type="tool")
    def execute_move(self, move_uci: str) -> tuple[bool, str]:
        instruction = f"Move piece from {move_uci[:2]} to {move_uci[2:4]}"
        return True, instruction
