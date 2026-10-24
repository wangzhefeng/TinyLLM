from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CausalLM(Protocol):
    def forward(
        self,
        input_ids: torch.Tensor,
        cache=None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def reset_kv_cache(self) -> None: ...


@runtime_checkable
class EncoderLM(Protocol):
    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class GenerationCache(Protocol):
    position: int

    def get(self, layer_idx: int): ...

    def update(self, layer_idx: int, value) -> None: ...

    def reset(self) -> None: ...
