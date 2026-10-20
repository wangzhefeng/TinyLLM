# -*- coding: utf-8 -*-

"""LoRA / DoRA 兼容 shim（2026-10-20 起规范实现在 layers/lora.py）。

保留历史导入路径 `post_training.peft.layers.lora_dora`，全部转发到 layers/lora.py。
"""

from layers.lora import (
    LinearWithDoRA,
    LinearWithDoRAMerged,
    LinearWithLoRA,
    LinearWithLoRAMerged,
    LoRALayer,
    freeze_linear_layers,
    merge_lora_weights,
    replace_linear_with_lora,
)

__all__ = [
    "LoRALayer",
    "LinearWithLoRA",
    "LinearWithDoRA",
    "LinearWithLoRAMerged",
    "LinearWithDoRAMerged",
    "replace_linear_with_lora",
    "freeze_linear_layers",
    "merge_lora_weights",
]
