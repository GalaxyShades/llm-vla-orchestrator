"""File logging and artefact generation for chess move orchestration."""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import chess
import chess.pgn

from orchestrator.chess_types import ChessMoveRecord

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")
LOGGER = logging.getLogger(__name__)


class ChessMoveLogger:
    """Writes detailed move artefacts, JSONL records, and board snapshots."""

    def __init__(self, base_dir: str = "games", game_id: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.game_id = game_id or datetime.now(HONG_KONG_TZ).strftime("%Y%m%d_%H%M%S")
        self._initialise_paths()

    def ensure_game(self, game_id: str) -> None:
        """Rotate logger paths if the game identity changed."""
        if game_id == self.game_id:
            return
        self.game_id = game_id
        self._initialise_paths()

    def _initialise_paths(self) -> None:
        self.game_dir = self.base_dir / self.game_id
        self.moves_dir = self.game_dir / "moves"
        self.game_dir.mkdir(parents=True, exist_ok=True)
        self.moves_dir.mkdir(parents=True, exist_ok=True)

        self.moves_path = self.game_dir / "moves.jsonl"
        self.pgn_path = self.game_dir / "game.pgn"
        self.ui_state_path = self.game_dir / "ui_state.json"
        self.run_dir = self.moves_dir

    def load_ui_state(self) -> dict:
        """Load persisted frontend UI state for the current game directory."""
        if not self.ui_state_path.exists():
            return {}
        with self.ui_state_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {}
        return data

    def save_ui_state(self, payload: dict) -> None:
        """Persist frontend UI state for reload sync."""
        with self.ui_state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def start_new_move(self, move_index: int) -> None:
        self.run_dir = self.moves_dir / f"move_{move_index:03d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.boards_dir = self.run_dir
        LOGGER.info("move_%03d: created move directory at %s", move_index, self.run_dir)

    def append_move(self, record: ChessMoveRecord) -> None:
        with self.moves_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def write_pgn(self, initial_fen: str, moves_uci: list[str]) -> None:
        board = chess.Board(initial_fen)
        for move_uci in moves_uci:
            board.push(chess.Move.from_uci(move_uci))

        game = chess.pgn.Game.from_board(board)
        if initial_fen != chess.STARTING_FEN:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = initial_fen

        with self.pgn_path.open("w", encoding="utf-8") as handle:
            print(game, file=handle, end="\n")

    def render_boards(self, *, move_index: int, pre_fen: str, post_fen: str, observed_piece_placement: str) -> None:
        stem = f"move_{move_index:03d}"
        pre_board = chess.Board(pre_fen)
        self._write_png(pre_board, self.boards_dir / f"{stem}_pre.png")

        post_board = chess.Board(post_fen)
        self._write_png(post_board, self.boards_dir / f"{stem}_post.png")

        observed_board = chess.Board(pre_fen)
        observed_board.set_board_fen(observed_piece_placement)
        self._write_png(observed_board, self.boards_dir / f"{stem}_observed.png")

    def save_pre_board(self, *, move_index: int, pre_fen: str) -> str:
        """Persist the board snapshot before player move application."""
        stem = f"move_{move_index:03d}"
        out_path = self.boards_dir / f"{stem}_pre.png"
        self._write_png(chess.Board(pre_fen), out_path)
        LOGGER.info("move_%03d: saved pre snapshot to %s", move_index, out_path)
        return str(out_path)

    def save_observed_board(
        self,
        *,
        move_index: int,
        pre_fen: str,
        observed_piece_placement: str,
    ) -> str:
        """Persist the observed board state after player move recognition."""
        stem = f"move_{move_index:03d}"
        out_path = self.boards_dir / f"{stem}_observed.png"
        observed_board = chess.Board(pre_fen)
        observed_board.set_board_fen(observed_piece_placement)
        self._write_png(observed_board, out_path)
        LOGGER.info("move_%03d: saved observed snapshot to %s", move_index, out_path)
        return str(out_path)

    def save_post_board(self, *, move_index: int, post_fen: str) -> str:
        """Persist the board snapshot after AI move application."""
        stem = f"move_{move_index:03d}"
        out_path = self.boards_dir / f"{stem}_post.png"
        self._write_png(chess.Board(post_fen), out_path)
        LOGGER.info("move_%03d: saved post snapshot to %s", move_index, out_path)
        return str(out_path)

    def save_analysis_input_image(
        self,
        *,
        move_index: int,
        source: str,
        image_data_url: str,
    ) -> str | None:
        """Persist the image sent for analysis and return its path."""
        if not image_data_url:
            return None
        if "," not in image_data_url:
            return None
        header, payload = image_data_url.split(",", 1)
        if ";base64" not in header:
            return None
        try:
            raw = base64.b64decode(payload)
        except Exception:  # noqa: BLE001
            return None
        safe_source = "".join(ch for ch in source if ch.isalnum() or ch in ("_", "-")) or "unknown"
        out_path = self.run_dir / f"move_{move_index:03d}_{safe_source}.png"
        out_path.write_bytes(raw)
        LOGGER.info("move_%03d: saved analysis input image to %s", move_index, out_path)
        return str(out_path)

    @staticmethod
    def _write_png(board: chess.Board, out_path: Path, size: int = 640) -> None:
        from PIL import Image, ImageDraw, ImageFont

        tile = size // 8
        width = tile * 8
        image = Image.new("RGB", (width, width), "#f0d9b5")
        draw = ImageDraw.Draw(image)

        light = "#f0d9b5"
        dark = "#b58863"
        text_colour_white = "#ffffff"
        text_colour_black = "#222222"
        font = ImageFont.load_default()

        for rank in range(8):
            for file in range(8):
                x0 = file * tile
                y0 = rank * tile
                fill = dark if (file + rank) % 2 else light
                draw.rectangle([x0, y0, x0 + tile, y0 + tile], fill=fill)

        for square, piece in board.piece_map().items():
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            x = file * tile + tile // 2
            y = (7 - rank) * tile + tile // 2
            symbol = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            colour = text_colour_white if piece.color == chess.WHITE else text_colour_black
            try:
                bbox = font.getbbox(symbol)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except AttributeError:
                tw, th = font.getsize(symbol)
            draw.text((x - tw / 2, y - th / 2), symbol, fill=colour, font=font)

        image.save(out_path, format="PNG")
