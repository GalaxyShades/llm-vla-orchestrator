"""Execution adapters for applying chess moves on external hardware."""

from __future__ import annotations

import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langsmith import traceable


class PiZeroExecutor:
    """Stub executor for Pi Zero move commands in development."""

    @traceable(name="chess_executor_execute_move", run_type="tool")
    def execute_move(self, move_uci: str) -> tuple[bool, str]:
        instruction = f"Move piece from {move_uci[:2]} to {move_uci[2:4]}"
        return True, instruction


@dataclass(frozen=True)
class CobotChessExecutorConfig:
    """Runtime settings for sending chess square poses to the cobot ROS bridge."""

    square_poses_path: Path
    cobot_module_path: Path
    active_arm: str = "left"
    move_method: str = "continuous"
    left_idle_pose: tuple[float, ...] | None = None
    right_idle_pose: tuple[float, ...] | None = None
    settle_s: float = 0.0


class CobotChessExecutor:
    """Execute chess moves by publishing calibrated joint targets to Cobot Magic."""

    def __init__(self, config: CobotChessExecutorConfig) -> None:
        self.config = config
        self.square_poses = self._load_square_poses(config.square_poses_path)
        self.robot_utils = self._load_robot_utils(config.cobot_module_path)
        self.args = self.robot_utils.get_arguments()
        self.ros_operator = self.robot_utils.RosOperator(self.args)

    @traceable(name="chess_cobot_executor_execute_move", run_type="tool")
    def execute_move(self, move_uci: str) -> tuple[bool, str]:
        source_square = move_uci[:2]
        target_square = move_uci[2:4]
        source_pose = self._pose_for_square(source_square)
        target_pose = self._pose_for_square(target_square)

        self._publish_pose(source_pose)
        self._settle()
        self._publish_pose(target_pose)
        self._settle()

        instruction = (
            f"Cobot moved {self.config.active_arm} arm from "
            f"{source_square} to {target_square}"
        )
        return True, instruction

    @staticmethod
    def _load_square_poses(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return dict(payload.get("squares", payload))

    @staticmethod
    def _load_robot_utils(module_path: Path) -> Any:
        package_root = module_path.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        examples_pkg = sys.modules.setdefault(
            "examples",
            types.ModuleType("examples"),
        )
        examples_pkg.__path__ = [str(package_root)]

        aloha_pkg = sys.modules.setdefault(
            "examples.mobile_aloha_AgileX",
            types.ModuleType("examples.mobile_aloha_AgileX"),
        )
        aloha_pkg.__path__ = [str(module_path)]

        return importlib.import_module("examples.mobile_aloha_AgileX.robot_utils")

    def _pose_for_square(self, square: str) -> tuple[float, ...]:
        entry = self.square_poses[square]
        if isinstance(entry, dict):
            pose = entry[self.config.active_arm]
        else:
            pose = entry
        return tuple(float(value) for value in pose)

    def _publish_pose(self, active_pose: tuple[float, ...]) -> None:
        left_pose, right_pose = self._paired_poses(active_pose)
        publish = getattr(
            self.ros_operator,
            f"puppet_arm_publish_{self.config.move_method}",
        )
        publish(list(left_pose), list(right_pose))

    def _paired_poses(
        self,
        active_pose: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.config.active_arm == "left":
            return active_pose, self._required_idle_pose("right")
        if self.config.active_arm == "right":
            return self._required_idle_pose("left"), active_pose
        raise ValueError(f"Unsupported active_arm: {self.config.active_arm}")

    def _required_idle_pose(self, arm: str) -> tuple[float, ...]:
        pose = (
            self.config.left_idle_pose
            if arm == "left"
            else self.config.right_idle_pose
        )
        if pose is None:
            raise ValueError(f"{arm}_idle_pose is required for cobot execution")
        return pose

    def _settle(self) -> None:
        if self.config.settle_s:
            time.sleep(self.config.settle_s)


def build_executor(execution_cfg: dict[str, Any] | None) -> PiZeroExecutor | CobotChessExecutor:
    """Build the configured hardware executor for the chess pipeline."""

    cfg = dict(execution_cfg or {})
    executor_type = str(cfg.get("type", "dry_run")).strip().lower()
    if executor_type in {"dry_run", "pi_zero", "stub"}:
        return PiZeroExecutor()
    if executor_type != "cobot":
        raise ValueError(f"Unknown execution.type: {executor_type}")

    module_path = Path(
        str(cfg.get("cobot_module_path", "cobot_magic/mobile_aloha_AgileX"))
    ).expanduser()
    square_poses_path = Path(str(cfg["square_poses_path"])).expanduser()
    config = CobotChessExecutorConfig(
        square_poses_path=square_poses_path,
        cobot_module_path=module_path,
        active_arm=str(cfg.get("active_arm", "left")).strip().lower(),
        move_method=str(cfg.get("move_method", "continuous")).strip().lower(),
        left_idle_pose=_optional_pose(cfg.get("left_idle_pose")),
        right_idle_pose=_optional_pose(cfg.get("right_idle_pose")),
        settle_s=float(cfg.get("settle_s", 0.0)),
    )
    return CobotChessExecutor(config)


def _optional_pose(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)
