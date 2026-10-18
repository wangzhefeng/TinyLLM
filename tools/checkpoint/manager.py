"""Checkpoint rotation and resume helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import latest_checkpoint, load_checkpoint, save_checkpoint


class CheckpointManager:
    """Save ``step_XXXXXXXX.pt`` files and keep only the newest N."""

    def __init__(
        self,
        directory: str | Path,
        keep_last: int = 3,
        prefix: str = "step",
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.directory = Path(directory).expanduser()
        self.keep_last = keep_last
        self.prefix = prefix
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, step: int) -> Path:
        return self.directory / f"{self.prefix}_{step:08d}.pt"

    def save(
        self,
        step: int,
        model: Any | None = None,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = save_checkpoint(
            self.path_for(step),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            extra=extra,
        )
        self._prune()
        return path

    def latest(self) -> Path | None:
        return latest_checkpoint(self.directory)

    def load_latest(self, **kwargs: Any) -> dict[str, Any] | None:
        path = self.latest()
        if path is None:
            return None
        return load_checkpoint(path, **kwargs)

    def _prune(self) -> None:
        paths = sorted(self.directory.glob(f"{self.prefix}_*.pt"))
        for old in paths[: -self.keep_last]:
            old.unlink(missing_ok=True)
