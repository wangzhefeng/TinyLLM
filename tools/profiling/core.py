"""Reusable profiling helpers for LLM training and inference."""

from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass
from typing import Iterator

import torch


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Return the number of parameters, optionally only trainable ones."""
    return sum(
        p.numel()
        for p in model.parameters()
        if not trainable_only or p.requires_grad
    )


def parameter_report(model: torch.nn.Module) -> dict[str, int]:
    """Return total/trainable/frozen parameter counts."""
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


@dataclass(frozen=True)
class ThroughputStats:
    tokens: int
    elapsed_s: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.tokens / self.elapsed_s

    def as_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "tokens_per_second": self.tokens_per_second,
        }


class ThroughputMeter:
    """Measure token throughput over an arbitrary code block."""

    def __init__(self, tokens: int = 0) -> None:
        self.tokens = tokens
        self._start: float | None = None
        self._elapsed_s = 0.0

    def start(self) -> "ThroughputMeter":
        self._start = time.perf_counter()
        return self

    def update(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        self.tokens += tokens

    def stop(self) -> ThroughputStats:
        if self._start is None:
            raise RuntimeError("ThroughputMeter.start() must be called first")
        self._elapsed_s = time.perf_counter() - self._start
        self._start = None
        return self.stats

    @property
    def elapsed_s(self) -> float:
        if self._start is not None:
            return time.perf_counter() - self._start
        return self._elapsed_s

    @property
    def stats(self) -> ThroughputStats:
        return ThroughputStats(tokens=self.tokens, elapsed_s=self.elapsed_s)


@contextlib.contextmanager
def measure_throughput(initial_tokens: int = 0) -> Iterator[ThroughputMeter]:
    """Context manager that yields a meter and stops it on exit.

    Example:
        with measure_throughput() as meter:
            ...
            meter.update(batch_size * seq_len)
        print(meter.stats.tokens_per_second)
    """
    meter = ThroughputMeter(initial_tokens).start()
    try:
        yield meter
    finally:
        meter.stop()


def memory_report(device: str | torch.device | None = None) -> dict[str, int | str]:
    """Return allocated/reserved memory for CUDA/MPS, or 0 for CPU."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    allocated = reserved = peak = 0
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        # MPS exposes allocated memory only; driver memory is optional.
        allocated = int(torch.mps.current_allocated_memory())
        reserved = allocated
        try:
            reserved = int(torch.mps.driver_allocated_memory())
        except (AttributeError, RuntimeError):
            pass
        peak = allocated

    return {
        "device": str(device),
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_bytes": peak,
    }


def estimate_transformer_flops(
    num_parameters: int,
    num_tokens: int,
    factor: float = 6.0,
) -> float:
    """Estimate training FLOPs with the standard ~6*N*T approximation."""
    if num_parameters < 0 or num_tokens < 0:
        raise ValueError("num_parameters and num_tokens must be >= 0")
    return factor * num_parameters * num_tokens


def estimate_mfu(achieved_flops: float, peak_flops: float) -> float:
    """Model FLOPs Utilization (0..1)."""
    if peak_flops <= 0:
        raise ValueError("peak_flops must be > 0")
    return achieved_flops / peak_flops


def human_bytes(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"
