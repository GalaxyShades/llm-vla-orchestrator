"""ChatGPT vision recogniser for extracting a SAN move from chess camera images."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

import chess
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel

from orchestrator.chess_types import VisionMoveState


class _VisionOutput(BaseModel):
    """Structured output schema for vision SAN move parsing."""

    after_piece_placement: str
    move_san: str
    overall_confidence: Optional[float] = None


class ChatGPTVisionRecognizer:
    """Uses ChatGPT vision to extract a SAN move in a strict JSON schema."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        api_version: str | None = None,
        azure_endpoint: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model.strip()
        self.max_retries = max(1, int(max_retries))
        if not self.model:
            raise ValueError("ChatGPTVisionRecognizer requires a non-empty model")
        if not api_key.strip():
            raise ValueError("ChatGPTVisionRecognizer requires a non-empty api_key")
        api_version_value = str(api_version or "").strip()
        azure_endpoint_value = str(azure_endpoint or "").strip()

        if api_version_value:
            if not azure_endpoint_value:
                raise ValueError(
                    "ChatGPTVisionRecognizer requires azure_endpoint when api_version is set"
                )
            self.llm = AzureChatOpenAI(
                azure_deployment=self.model,
                api_key=api_key,
                api_version=api_version_value,
                azure_endpoint=azure_endpoint_value,
            )
        else:
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                base_url=base_url,
            )

    @traceable(name="chess_pipeline_vision_infer_san", run_type="tool")
    def recognise_move(
        self,
        *,
        image_path: str,
        before_fen: str,
        feedback: str | None = None,
    ) -> tuple[VisionMoveState, int]:
        raw_bytes = Path(image_path).read_bytes()

        attempts = 0
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                messages = [
                    SystemMessage(
                        content=(
                            "You are a strict chessboard parser. You are given the board state before a "
                            "move and a top-down image of the board after that move.\n\n"
                            "Step 1: infer the full after-position from the image.\n"
                            "Step 2: compare the before-position and after-position and determine the "
                            "single legal move.\n\n"
                            "Return strict JSON with keys:\n"
                            "- after_piece_placement\n"
                            "- move_san\n"
                            "- overall_confidence\n\n"
                            "after_piece_placement must be valid board-FEN placement only.\n"
                            "move_san must be standard algebraic notation only.\n"
                            "overall_confidence must be a number between 0 and 1 or null.\n"
                            "Do not include explanations."
                        )
                    ),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Task: identify the move in algebraic notation (SAN). "
                                    f"before_fen: {before_fen}"
                                ),
                            },
                            *(
                                [{"type": "text", "text": f"Feedback from previous attempt: {feedback}"}]
                                if feedback
                                else []
                            ),
                            {"type": "image_url", "image_url": {"url": self._to_data_url(raw_bytes)}},
                        ]
                    ),
                ]
                llm_response = self.llm.invoke(messages)
                raw_text = str(getattr(llm_response, "content", "")).strip()
                payload = _VisionOutput.model_validate(
                    self._normalise_payload(self._parse_json_object(raw_text))
                )
                raw_output = payload.model_dump_json()
                after_piece_placement = self._normalise_piece_placement(
                    str(payload.after_piece_placement).strip()
                )
                result = VisionMoveState(
                    after_piece_placement=after_piece_placement,
                    move_san=str(payload.move_san).strip(),
                    overall_confidence=(
                        float(payload.overall_confidence)
                        if payload.overall_confidence is not None
                        else None
                    ),
                    raw_model_output=raw_output,
                )
                return result, attempts
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f"Vision recognition failed after {attempts} attempts: {last_error}") from last_error

    @staticmethod
    def _to_data_url(raw: bytes) -> str:
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ValueError("empty vision response")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Could not parse JSON object from vision response")

    @staticmethod
    def _normalise_payload(payload: dict[str, Any]) -> dict[str, Any]:
        normalised = dict(payload)

        after_piece_placement = normalised.get("after_piece_placement")
        if not isinstance(after_piece_placement, str):
            for alias in ("piece_placement", "after_fen", "post_piece_placement"):
                alias_value = normalised.get(alias)
                if isinstance(alias_value, str) and alias_value.strip():
                    after_piece_placement = alias_value
                    break
        normalised["after_piece_placement"] = str(after_piece_placement or "").strip()

        move_san = normalised.get("move_san")
        if not isinstance(move_san, str):
            for alias in ("san", "move", "move_notation", "move_algebraic"):
                alias_value = normalised.get(alias)
                if isinstance(alias_value, str) and alias_value.strip():
                    move_san = alias_value
                    break
        normalised["move_san"] = str(move_san or "").strip()

        overall_confidence = normalised.get("overall_confidence")
        if isinstance(overall_confidence, str):
            aliases = {
                "very_low": 0.1,
                "low": 0.25,
                "medium": 0.5,
                "high": 0.75,
                "very_high": 0.9,
            }
            mapped = aliases.get(overall_confidence.strip().lower().replace(" ", "_"))
            normalised["overall_confidence"] = mapped
        elif overall_confidence is None:
            normalised["overall_confidence"] = None
        else:
            try:
                normalised["overall_confidence"] = float(overall_confidence)
            except (TypeError, ValueError):
                normalised["overall_confidence"] = None

        return normalised

    @staticmethod
    def _normalise_piece_placement(raw_value: str) -> str:
        value = str(raw_value).strip()
        if not value:
            raise ValueError("empty after_piece_placement in vision response")
        if " " in value:
            return chess.Board(value).board_fen()
        chess.Board(f"{value} w - - 0 1")
        return value
