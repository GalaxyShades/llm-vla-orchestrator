"""LLM-backed policy agent that selects a move from engine candidates."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langsmith import traceable

from orchestrator.chess_types import ChessOrchestratorDecision, EngineCandidate


class ChessOrchestratorAgent:
    """Chooses one move from top engine candidates using an LLM policy call."""

    def __init__(
        self,
        *,
        candidate_count: int = 5,
        objective_prompt: str = "",
        model: str,
        api_key: str,
        base_url: str | None = None,
        api_version: str | None = None,
        azure_endpoint: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.candidate_count = max(2, int(candidate_count))
        self.objective_prompt = objective_prompt
        self.model = model.strip()
        self.max_retries = max(1, int(max_retries))
        if not self.model:
            raise ValueError("ChessOrchestratorAgent requires a non-empty model")
        if not api_key.strip():
            raise ValueError("ChessOrchestratorAgent requires a non-empty api_key")
        api_version_value = str(api_version or "").strip()
        azure_endpoint_value = str(azure_endpoint or "").strip()

        if api_version_value:
            if not azure_endpoint_value:
                raise ValueError(
                    "ChessOrchestratorAgent requires azure_endpoint when api_version is set"
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

    @traceable(name="chess_orchestrator_agent_choose_move", run_type="tool")
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
        player_move_evidence: dict[str, Any] | None,
    ) -> ChessOrchestratorDecision:
        if not candidates:
            raise ValueError("ChessOrchestratorAgent requires at least one candidate move")
        shortlist = candidates[: self.candidate_count]
        candidate_map = {candidate.uci: candidate for candidate in shortlist}

        context = {
            "best_eval_cp": best_eval_cp,
            "player_estimated_elo": player_estimated_elo,
            "policy_mode": policy_mode,
            "game_objective": game_objective,
            "close_game_eval_window_cp": close_game_eval_window_cp,
            "target_cp_loss": target_cp_loss,
            "target_player_win_rate": target_player_win_rate,
            "allow_best_play": allow_best_play,
            "player_move_evidence": player_move_evidence,
            "objective_prompt": self.objective_prompt,
            "candidates": [
                {
                    "uci": candidate.uci,
                    "san": candidate.san,
                    "eval_cp": candidate.eval_cp,
                    "cp_loss": candidate.cp_loss,
                }
                for candidate in shortlist
            ],
        }
        guidance = (
            "You are the Chess Trainer. Choose exactly one move from candidates. "
            "Primary rule: keep the game close and competitive. Prefer moves that keep evaluation "
            "within close_game_eval_window_cp of equality when plausible. "
            "Secondary rule: apply game_objective as a bias, not an excuse to make the game one-sided. "
            "ai_should_win means slightly stronger practical pressure; ai_should_lose means slightly "
            "more training-friendly pressure. "
            "If the player blundered hard and allow_best_play is true, you may convert more directly."
        )

        feedback: str | None = None
        last_error = "unknown"
        for _ in range(self.max_retries):
            @tool
            def describe_candidate(uci: str) -> str:
                """Get full candidate details by UCI from the provided shortlist."""
                candidate = candidate_map.get(uci)
                if candidate is None:
                    return json.dumps({"error": f"Unknown candidate UCI: {uci}"})
                return json.dumps(
                    {
                        "uci": candidate.uci,
                        "san": candidate.san,
                        "eval_cp": candidate.eval_cp,
                        "cp_loss": candidate.cp_loss,
                    }
                )

            tools = [describe_candidate]
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            f"{guidance} "
                            "You may call tools to inspect candidates. "
                            "Return strict JSON with keys: selected_uci, reason, candidate_scores."
                        ),
                    ),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.llm, tools, prompt)
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
            )
            result = executor.invoke(
                {
                    "input": json.dumps(
                        {
                            "context": context,
                            "feedback": feedback,
                        }
                    )
                }
            )
            raw_output = str(result.get("output", "")).strip()
            payload = self._parse_json_object(raw_output)
            selected_uci = str(payload.get("selected_uci", "")).strip()
            if selected_uci not in candidate_map:
                last_error = f"selected_uci '{selected_uci}' is not one of the provided candidates"
                feedback = last_error
                continue
            selected = candidate_map[selected_uci]
            reason = str(payload.get("reason", "")).strip() or "LLM selected candidate move."
            raw_scores = payload.get("candidate_scores", {})
            candidate_scores = self._normalise_candidate_scores(
                raw_scores=raw_scores,
                shortlist=shortlist,
            )
            return ChessOrchestratorDecision(
                selected=selected,
                reason=reason,
                candidate_scores=candidate_scores,
            )

        raise RuntimeError(f"ChessOrchestratorAgent failed to return a valid candidate: {last_error}")

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ValueError("empty policy response")
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
        raise ValueError("Could not parse JSON object from policy response")

    @staticmethod
    def _normalise_candidate_scores(
        *,
        raw_scores: Any,
        shortlist: list[EngineCandidate],
    ) -> dict[str, float]:
        candidate_scores = {candidate.uci: 0.0 for candidate in shortlist}

        if isinstance(raw_scores, dict):
            for candidate in shortlist:
                try:
                    candidate_scores[candidate.uci] = float(raw_scores.get(candidate.uci, 0.0))
                except (TypeError, ValueError):
                    candidate_scores[candidate.uci] = 0.0
            return candidate_scores

        if isinstance(raw_scores, list):
            for item in raw_scores:
                if not isinstance(item, dict):
                    continue
                uci = str(item.get("uci", "")).strip()
                if uci not in candidate_scores:
                    continue
                raw_score = item.get("score", item.get("value", item.get("weight", 0.0)))
                try:
                    candidate_scores[uci] = float(raw_score)
                except (TypeError, ValueError):
                    continue

        return candidate_scores
