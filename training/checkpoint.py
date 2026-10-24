from __future__ import annotations

from pathlib import Path

from tools.checkpoint.io import load_checkpoint, save_checkpoint


def save_training_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    step: int = 0,
    config=None,
    extra=None,
):
    metadata = dict(extra or {})
    metadata["epoch"] = epoch
    if config is not None:
        metadata["config"] = config.to_dict() if hasattr(config, "to_dict") else config
    return save_checkpoint(
        Path(path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        extra=metadata,
    )


def load_training_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    payload = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=map_location,
        strict=strict,
    )
    payload.setdefault("epoch", payload.get("extra", {}).get("epoch", 0))
    payload.setdefault("config", payload.get("extra", {}).get("config"))
    return payload
