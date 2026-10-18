"""Atomic checkpoint save/load helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model/optimizer/scheduler state to a single checkpoint file.

    The file is written to a temporary path first, then atomically replaced.
    """
    path = Path(path).expanduser()
    payload: dict[str, Any] = {"step": int(step), "extra": extra or {}}
    if model is not None:
        payload["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    _atomic_torch_save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer/scheduler."""
    path = Path(path).expanduser()
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        payload = {"model_state_dict": payload}

    if model is not None:
        state = payload.get("model_state_dict", payload.get("model"))
        if state is None:
            raise KeyError(f"checkpoint has no model state: {path}")
        model.load_state_dict(state, strict=strict)

    if optimizer is not None:
        state = payload.get("optimizer_state_dict")
        if state is None:
            raise KeyError(f"checkpoint has no optimizer state: {path}")
        optimizer.load_state_dict(state)

    if scheduler is not None:
        state = payload.get("scheduler_state_dict")
        if state is None:
            raise KeyError(f"checkpoint has no scheduler state: {path}")
        scheduler.load_state_dict(state)

    return payload


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the most recently modified .pt/.pth file, or None."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return None
    paths = [
        *directory.glob("*.pt"),
        *directory.glob("*.pth"),
    ]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)
