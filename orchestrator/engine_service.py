"""Stockfish-backed chess engine analysis services."""

from __future__ import annotations

import chess
import chess.engine
from langsmith import traceable

from orchestrator.chess_types import EngineCandidate, EngineMoveSelection, PlayerMoveEvidence


class StockfishService:
    """Wraps local Stockfish analysis through python-chess."""

    def __init__(self, *, engine_path: str, think_time_s: float = 10.0, multipv: int = 8) -> None:
        self.engine_path = engine_path
        self.think_time_s = float(think_time_s)
        self.multipv = max(2, int(multipv))

    @traceable(name="chess_stockfish_analyse_move_quality", run_type="tool")
    def analyse_move_quality(self, board_before: chess.Board, move: chess.Move) -> PlayerMoveEvidence:
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            info_best = engine.analyse(board_before, chess.engine.Limit(time=self.think_time_s))
            eval_before = self._score_to_cp(info_best["score"], board_before.turn)

            board_after = board_before.copy(stack=False)
            move_san = board_before.san(move)
            board_after.push(move)
            info_after = engine.analyse(board_after, chess.engine.Limit(time=self.think_time_s))
            eval_after = self._score_to_cp(info_after["score"], board_after.turn)
            # Eval after is from side-to-move perspective, so invert to compare from player perspective.
            eval_after_player_perspective = -eval_after
            cpl = max(0, eval_before - eval_after_player_perspective)

        return PlayerMoveEvidence(
            move_uci=move.uci(),
            move_san=move_san,
            eval_before_cp=eval_before,
            eval_after_cp=eval_after_player_perspective,
            centipawn_loss=cpl,
            player_time_s=None,
        )

    @traceable(name="chess_stockfish_get_top_move_candidates", run_type="tool")
    def get_top_move_candidates(self, *, board: chess.Board, top_k: int = 5) -> EngineMoveSelection:
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            infos = engine.analyse(
                board,
                chess.engine.Limit(time=self.think_time_s),
                multipv=max(2, min(self.multipv, int(top_k))),
            )

        sorted_candidates: list[EngineCandidate] = []
        best_eval: int | None = None

        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue
            move = pv[0]
            eval_cp = self._score_to_cp(score, board.turn)
            if best_eval is None:
                best_eval = eval_cp
            san = board.san(move)
            cp_loss = max(0, (best_eval or eval_cp) - eval_cp)
            sorted_candidates.append(
                EngineCandidate(
                    uci=move.uci(),
                    san=san,
                    eval_cp=eval_cp,
                    cp_loss=cp_loss,
                )
            )

        if not sorted_candidates:
            raise RuntimeError("Stockfish did not return any candidates")

        best_candidate = sorted_candidates[0]
        best_eval = best_candidate.eval_cp

        # Recompute cp loss against fixed best eval in case multipv ordering is not exact.
        for idx, candidate in enumerate(sorted_candidates):
            sorted_candidates[idx] = EngineCandidate(
                uci=candidate.uci,
                san=candidate.san,
                eval_cp=candidate.eval_cp,
                cp_loss=max(0, best_eval - candidate.eval_cp),
            )

        return EngineMoveSelection(
            best_eval_cp=best_eval,
            selected=sorted_candidates[0],
            candidates=sorted_candidates,
        )

    @staticmethod
    def _score_to_cp(score: chess.engine.PovScore, pov_colour: bool) -> int:
        cp = score.pov(pov_colour).score(mate_score=10000)
        if cp is None:
            return 0
        return int(cp)
