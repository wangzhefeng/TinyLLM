from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch


class ConfigMixin:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self):
        return type(self).from_dict(self.to_dict())

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | Any):
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            values = vars(values)
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


@dataclass
class ModelConfig(ConfigMixin):
    vocab_size: int
    context_length: int
    embed_dim: int
    n_heads: int
    n_layers: int
    d_ff: int | None = None
    dropout: float = 0.0
    dtype: torch.dtype = torch.float32
    head_dim: int | None = None
    n_kv_groups: int | None = None
    qkv_bias: bool = True
    qk_norm: bool = False
    rope_base: float = 10_000.0
    rope_freq: dict | None = None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int | None = None
    tie_word_embeddings: bool = False
    kv_sharing_span: int = 1
    partial_rotary_factor: float = 1.0
    # 混合注意力（Olmo 3 滑窗/full 间隔层；Qwen3.5 linear/full 间隔层）
    sliding_window: int | None = None
    full_attention_interval: int = 0
    # Qwen3.5 full-attention 层的 sigmoid 输出门控
    attn_output_gate: bool = False
    # Qwen3.5 linear-attention（gated delta net）层
    linear_num_key_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_num_value_heads: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int = 4

    @classmethod
    def from_dict(cls, values):
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            values = vars(values)
        values = dict(values)
        if "d_ff" not in values and "hidden_dim" in values:
            values["d_ff"] = values["hidden_dim"]
        return super().from_dict(values)

    @property
    def hidden_dim(self) -> int | None:
        return self.d_ff

    @hidden_dim.setter
    def hidden_dim(self, value: int | None) -> None:
        self.d_ff = value


@dataclass
class TrainConfig(ConfigMixin):
    batch_size: int = 1
    learning_rate: float = 5e-4
    train_epochs: int = 1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    use_amp: bool = False
    seed: int = 42


@dataclass
class GenerationConfig(ConfigMixin):
    max_new_tokens: int = 50
    context_length: int | None = None
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    use_cache: bool = True
    draft_model_key: str | None = None
