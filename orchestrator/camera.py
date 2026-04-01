"""Camera input abstractions for chessboard image capture."""

from __future__ import annotations

from pathlib import Path


class DirectoryCamera:
    """Reads the latest chessboard image from a configured directory."""

    VALID_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(self, inbox_dir: str, current_filename: str = "camera_capture.jpg") -> None:
        self.inbox_dir = Path(inbox_dir)
        self.current_filename = current_filename

    def get_current_image(self) -> Path:
        explicit = self.inbox_dir / self.current_filename
        if explicit.exists():
            return explicit

        stem = Path(self.current_filename).stem
        for suffix in self.VALID_SUFFIXES:
            candidate = self.inbox_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

        candidates = [
            path
            for path in self.inbox_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.VALID_SUFFIXES
        ]
        if not candidates:
            raise ValueError(f"No image found in inbox directory: {self.inbox_dir}")
        return max(candidates, key=lambda path: path.stat().st_mtime)
