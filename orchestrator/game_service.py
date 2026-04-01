"""Core chess game service for validation, policy, execution, and logging."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import logging
from typing import Any

import chess

from orchestrator.chess_types import ChessMoveRecord, ChessOrchestratorDecision, TransitionValidation
from orchestrator.difficulty import DifficultyController
from orchestrator.engine_service import StockfishService
from orchestrator.executor import PiZeroExecutor
from orchestrator.game_logger import ChessMoveLogger
from orchestrator.game_state import HONG_KONG_TZ, ChessMemoryStore
from orchestrator.policy_agent import ChessOrchestratorAgent

LOGGER = logging.getLogger(__name__)


class ChessGameService:
    """Coordinates player move validation, adaptive AI move choice, execution, and logging."""

    def __init__(
        self,
        *,
        memory_store: ChessMemoryStore,
        logger: ChessMoveLogger,
        stockfish_service: StockfishService,
        chess_orchestrator_agent: ChessOrchestratorAgent,
        difficulty_controller: DifficultyController,
        executor: PiZeroExecutor,
        player_colour: str = "white",
        assume_legal_player: bool = True,
    ) -> None:
        self.memory_store = memory_store
        self.logger = logger
        self.stockfish_service = stockfish_service
        self.chess_orchestrator_agent = chess_orchestrator_agent
        self.difficulty_controller = difficulty_controller
        self.executor = executor
        self.player_colour = player_colour
        self.assume_legal_player = assume_legal_player

    def move(
        self,
        *,
        observed_piece_placement: str,
        player_time_s: float | None,
        override_illegal: bool = False,
        source: str = "simulated",
        vision_attempts_used: int = 1,
        analysis_image_data_url: str | None = None,
        view_mode: str | None = None,
        camera_pitch_deg: float | None = None,
        camera_distance: float | None = None,
    ) -> dict[str, Any]:
        LOGGER.info("move request received: source=%s", source)
        state = self.memory_store.load()
        self.logger.ensure_game(str(state["game_id"]))
        memory = state["memory"]
        stats = memory["stats"]
        metadata = memory.setdefault("metadata", {})

        game_objective = str(metadata.get("game_objective", "")).strip()
        if game_objective not in {"ai_should_win", "ai_should_lose"}:
            game_objective = self.difficulty_controller.sample_game_objective()
            metadata["game_objective"] = game_objective
            metadata["game_objective_set_at"] = datetime.now(HONG_KONG_TZ).isoformat()

        pre_fen = str(state["current_fen"])
        board_before = chess.Board(pre_fen)

        LOGGER.info(
            "move_%03d: validating observed board transition",
            int(state["move_index"]) + 1,
        )
        validation = self._validate_transition(
            board_before=board_before,
            observed_piece_placement=observed_piece_placement,
        )

        if not validation.is_legal and not override_illegal and self.assume_legal_player:
            inferred = self._infer_most_likely_legal_transition(
                board_before=board_before,
                observed_piece_placement=observed_piece_placement,
            )
            if inferred is not None:
                validation = inferred
                observed_piece_placement = self._board_after_move_piece_placement(
                    board_before=board_before,
                    move_uci=str(inferred.matched_move_uci),
                )

        prospective_move_index = int(state["move_index"]) + 1
        self.logger.start_new_move(prospective_move_index)
        self.logger.save_pre_board(move_index=prospective_move_index, pre_fen=pre_fen)
        analysis_image_path = self.logger.save_analysis_input_image(
            move_index=prospective_move_index,
            source=source,
            image_data_url=analysis_image_data_url or "",
        )
        self.logger.save_observed_board(
            move_index=prospective_move_index,
            pre_fen=pre_fen,
            observed_piece_placement=observed_piece_placement,
        )

        if not validation.is_legal and not override_illegal:
            LOGGER.warning(
                "move_%03d: illegal transition detected (%s)",
                prospective_move_index,
                validation.warning,
            )
            pending = {
                "observed_piece_placement": observed_piece_placement,
                "player_time_s": player_time_s,
                "source": source,
                "warning": validation.warning,
                "created_at": datetime.now(HONG_KONG_TZ).isoformat(),
                "analysis_image_path": analysis_image_path,
                "view_mode": view_mode,
                "camera_pitch_deg": camera_pitch_deg,
                "camera_distance": camera_distance,
            }
            state["pending_illegal_transition"] = pending
            stats["illegal_moves"] = int(stats.get("illegal_moves", 0)) + 1
            memory["events"].append(
                {
                    "timestamp": pending["created_at"],
                    "event_type": "illegal_transition_detected",
                    "move_index": int(state["move_index"]) + 1,
                    "warning": validation.warning,
                }
            )
            self.memory_store.save(state)
            return {
                "status": "illegal_transition_warning",
                "warning": validation.warning,
                "can_override": True,
                "move_index": prospective_move_index,
                "analysis_image_path": analysis_image_path,
                "view_mode": view_mode,
                "camera_pitch_deg": camera_pitch_deg,
                "camera_distance": camera_distance,
            }

        if not validation.is_legal and override_illegal:
            LOGGER.info("move_%03d: applying override for illegal transition", prospective_move_index)
            ai_colour = chess.BLACK if self.player_colour.lower() == "white" else chess.WHITE
            board_after_player = chess.Board(pre_fen)
            board_after_player.set_board_fen(observed_piece_placement)
            board_after_player.turn = ai_colour
            player_move_evidence = None
            player_move_uci = None
            player_move_san = None
            override_used = True
            stats["overrides"] = int(stats.get("overrides", 0)) + 1
        else:
            assert validation.matched_move_uci is not None
            LOGGER.info(
                "move_%03d: accepted player move %s (%s)",
                prospective_move_index,
                validation.matched_move_uci,
                validation.matched_move_san,
            )
            matched_move = chess.Move.from_uci(validation.matched_move_uci)
            player_move_evidence = self.stockfish_service.analyse_move_quality(board_before, matched_move)
            player_move_evidence.player_time_s = player_time_s
            board_after_player = board_before.copy(stack=False)
            board_after_player.push(matched_move)
            player_move_uci = validation.matched_move_uci
            player_move_san = validation.matched_move_san
            override_used = False

        player_history = list(state.get("player_history", []))
        if player_move_evidence is not None:
            player_history.append(asdict(player_move_evidence))
        state["player_history"] = player_history

        player_estimated_elo = self.difficulty_controller.estimate_player_elo(player_history)
        effective_game_objective, objective_reason = self.difficulty_controller.resolve_effective_objective(
            game_objective=game_objective,
            player_history=player_history,
            latest_player_move_evidence=asdict(player_move_evidence) if player_move_evidence else None,
        )
        policy_mode = self.difficulty_controller.choose_policy_mode(effective_game_objective)
        target_cp_loss = self.difficulty_controller.target_cp_loss(policy_mode)

        allow_best_play = False
        if player_move_evidence is not None:
            allow_best_play = (
                player_move_evidence.centipawn_loss
                >= self.difficulty_controller.config.allow_conversion_after_player_blunder_cp
            )

        candidate_pack = self.stockfish_service.get_top_move_candidates(
            board=board_after_player,
            top_k=self.chess_orchestrator_agent.candidate_count,
        )
        LOGGER.info(
            "move_%03d: stockfish produced %d candidates",
            prospective_move_index,
            len(candidate_pack.candidates),
        )
        orchestrator_decision: ChessOrchestratorDecision = self.chess_orchestrator_agent.choose_move(
            candidates=candidate_pack.candidates,
            best_eval_cp=candidate_pack.best_eval_cp,
            player_estimated_elo=player_estimated_elo,
            policy_mode=policy_mode,
            game_objective=effective_game_objective,
            close_game_eval_window_cp=self.difficulty_controller.config.close_game_eval_window_cp,
            target_cp_loss=target_cp_loss,
            target_player_win_rate=self.difficulty_controller.config.target_player_win_rate,
            allow_best_play=allow_best_play,
            player_move_evidence=asdict(player_move_evidence) if player_move_evidence else None,
        )

        selected_candidate = orchestrator_decision.selected
        LOGGER.info(
            "move_%03d: orchestrator selected AI move %s (%s)",
            prospective_move_index,
            selected_candidate.uci,
            selected_candidate.san,
        )
        ai_move = chess.Move.from_uci(selected_candidate.uci)
        board_after_ai = board_after_player.copy(stack=False)
        board_after_ai.push(ai_move)

        execution_ok, pi_instruction = self.executor.execute_move(selected_candidate.uci)
        if not execution_ok:
            raise RuntimeError("Pi Zero execution failed")
        LOGGER.info("move_%03d: executor completed (%s)", prospective_move_index, pi_instruction)

        move_index = int(state["move_index"]) + 1
        state["move_index"] = move_index
        state["current_fen"] = board_after_ai.fen()
        if player_move_uci:
            state["moves_uci"].append(player_move_uci)
        state["moves_uci"].append(selected_candidate.uci)
        state["pending_illegal_transition"] = None

        stats["total_moves"] = int(stats.get("total_moves", 0)) + 1
        stats["legal_moves"] = int(stats.get("legal_moves", 0)) + 1

        event_ts = datetime.now(HONG_KONG_TZ).isoformat()
        memory["journal"].append(
            {
                "timestamp": event_ts,
                "move_index": move_index,
                "status": "ok",
                "player_move_uci": player_move_uci,
                "ai_move_uci": selected_candidate.uci,
                "policy_mode": policy_mode,
                "game_objective": game_objective,
                "effective_game_objective": effective_game_objective,
                "player_estimated_elo": player_estimated_elo,
            }
        )
        memory["events"].append(
            {
                "timestamp": event_ts,
                "event_type": "move_completed",
                "move_index": move_index,
                "source": source,
            }
        )
        memory["metadata"]["last_run_dir"] = str(self.logger.run_dir)

        self.memory_store.save(state)
        LOGGER.info("move_%03d: state saved", move_index)
        self.logger.write_pgn(initial_fen=str(state["initial_fen"]), moves_uci=state["moves_uci"])
        LOGGER.info("move_%03d: PGN updated at %s", move_index, self.logger.pgn_path)

        policy_context = {
            "game_objective": game_objective,
            "effective_game_objective": effective_game_objective,
            "objective_reason": objective_reason,
            "target_player_win_rate": self.difficulty_controller.config.target_player_win_rate,
            "close_game_eval_window_cp": self.difficulty_controller.config.close_game_eval_window_cp,
            "target_cp_loss": target_cp_loss,
            "allow_best_play": allow_best_play,
            "soft_mode_ratio": self.difficulty_controller.config.soft_mode_ratio,
        }
        stockfish_context = {
            "best_eval_cp": candidate_pack.best_eval_cp,
            "selected": asdict(selected_candidate),
            "candidates": [asdict(candidate) for candidate in candidate_pack.candidates],
            "orchestrator_decision_reason": orchestrator_decision.reason,
            "orchestrator_candidate_scores": orchestrator_decision.candidate_scores,
        }

        record = ChessMoveRecord(
            timestamp=event_ts,
            move_index=move_index,
            status="ok",
            pre_fen=pre_fen,
            observed_piece_placement=observed_piece_placement,
            player_move_uci=player_move_uci,
            ai_move_uci=selected_candidate.uci,
            ai_move_san=selected_candidate.san,
            post_fen=board_after_ai.fen(),
            warning=validation.warning,
            override_used=override_used,
            vision_attempts_used=vision_attempts_used,
            player_move_evidence=asdict(player_move_evidence) if player_move_evidence else None,
            player_estimated_elo=player_estimated_elo,
            policy_mode=policy_mode,
            policy_context=policy_context,
            stockfish_context=stockfish_context,
            pi_instruction=pi_instruction,
            execution_verified=True,
            analysis_image_path=analysis_image_path,
            logs={
                "source": source,
                "pending_override_was_used": override_used,
                "view_mode": view_mode,
                "camera_pitch_deg": camera_pitch_deg,
                "camera_distance": camera_distance,
            },
        )
        self.logger.append_move(record)
        LOGGER.info("move_%03d: move record appended to %s", move_index, self.logger.moves_path)
        self.logger.save_post_board(move_index=move_index, post_fen=board_after_ai.fen())
        LOGGER.info("move_%03d: move processing complete", move_index)

        return {
            "status": "ok",
            "move_index": move_index,
            "player_move_uci": player_move_uci,
            "player_move_san": player_move_san,
            "ai_move_uci": selected_candidate.uci,
            "ai_move_san": selected_candidate.san,
            "pi_instruction": pi_instruction,
            "player_estimated_elo": player_estimated_elo,
            "policy_mode": policy_mode,
            "game_objective": game_objective,
            "effective_game_objective": effective_game_objective,
            "policy_context": policy_context,
            "ai_reason": orchestrator_decision.reason,
            "warning": validation.warning,
            "run_dir": str(self.logger.run_dir),
            "moves_log": str(self.logger.moves_path),
            "pgn_path": str(self.logger.pgn_path),
            "boards_dir": str(self.logger.boards_dir),
            "post_fen": board_after_ai.fen(),
            "analysis_image_path": analysis_image_path,
            "view_mode": view_mode,
            "camera_pitch_deg": camera_pitch_deg,
            "camera_distance": camera_distance,
        }

    def override_move(self) -> dict[str, Any]:
        state = self.memory_store.load()
        pending = state.get("pending_illegal_transition")
        if not isinstance(pending, dict):
            raise ValueError("No pending illegal transition to override")

        return self.move(
            observed_piece_placement=str(pending["observed_piece_placement"]),
            player_time_s=pending.get("player_time_s"),
            override_illegal=True,
            source=str(pending.get("source", "simulated")),
            vision_attempts_used=1,
            analysis_image_data_url=None,
            view_mode=pending.get("view_mode"),
            camera_pitch_deg=pending.get("camera_pitch_deg"),
            camera_distance=pending.get("camera_distance"),
        )

    @staticmethod
    def _validate_transition(board_before: chess.Board, observed_piece_placement: str) -> TransitionValidation:
        for move in board_before.legal_moves:
            trial = board_before.copy(stack=False)
            trial.push(move)
            if trial.board_fen() == observed_piece_placement:
                return TransitionValidation(
                    is_legal=True,
                    matched_move_uci=move.uci(),
                    matched_move_san=board_before.san(move),
                    warning=None,
                )

        return TransitionValidation(
            is_legal=False,
            matched_move_uci=None,
            matched_move_san=None,
            warning=(
                "Observed board does not match any legal transition from the previous accepted state."
            ),
        )

    @classmethod
    def _infer_most_likely_legal_transition(
        cls,
        *,
        board_before: chess.Board,
        observed_piece_placement: str,
    ) -> TransitionValidation | None:
        observed_map = cls._piece_map_from_piece_placement(observed_piece_placement)
        best_move: chess.Move | None = None
        best_score = float("-inf")

        for move in board_before.legal_moves:
            trial = board_before.copy(stack=False)
            trial.push(move)
            trial_map = cls._piece_map_from_piece_placement(trial.board_fen())

            score = 0
            for square, piece_code in trial_map.items():
                if observed_map.get(square) == piece_code:
                    score += 1
            for square, piece_code in observed_map.items():
                if trial_map.get(square) == piece_code:
                    score += 1

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            return None

        return TransitionValidation(
            is_legal=True,
            matched_move_uci=best_move.uci(),
            matched_move_san=board_before.san(best_move),
            warning=(
                "Observed board was inconsistent with legal transitions; "
                "the most likely legal move was inferred."
            ),
        )

    @staticmethod
    def _piece_map_from_piece_placement(piece_placement: str) -> dict[str, str]:
        board = chess.Board(f"{piece_placement} w - - 0 1")
        out: dict[str, str] = {}
        for square, piece in board.piece_map().items():
            prefix = "w" if piece.color == chess.WHITE else "b"
            out[chess.square_name(square)] = f"{prefix}{piece.symbol().upper()}"
        return out

    @staticmethod
    def _board_after_move_piece_placement(*, board_before: chess.Board, move_uci: str) -> str:
        board = board_before.copy(stack=False)
        board.push(chess.Move.from_uci(move_uci))
        return board.board_fen()
