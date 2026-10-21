# -*- coding: utf-8 -*-

# ***************************************************
# * File        : clf_head.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 1.1.102121
# * Description : GPT 分类微调头：按 finetune_method 提供四种冻结/替换策略。
# *               - last_layer（默认，书 ch06 §6.4 做法）：冻结全部，只训新 out_head
# *               - last_transformer_stage（书 ch06 §6.5.2 实验）：额外解冻
# *                 最后一个 transformer block + final_norm
# *               - all_layers：全量微调 + 新 out_head
# *               - lora（书附录 E）：冻结全部，out_head 全量可训（不包 LoRA），
# *                 其余 Linear 加 LoRA 旁路（exclude out_head）
# * Link        : LLMs-from-scratch ch06 / 附录 E
# * Requirement : torch
# ***************************************************

__all__ = [
    "finetune_model",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch.nn as nn

from layers.lora import replace_linear_with_lora
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def finetune_model(model, emb_dim: int, num_classes: int,
                   finetune_method: str = "last_layer",
                   lora_rank: int = 16, lora_alpha: int = 16):
    """按冻结/替换策略改造 GPT 模型用于分类（替换 out_head 为 num_classes 类）。

    策略（对应 LLMs-from-scratch 的实验线）：
      - "last_layer"：冻结全部基座，仅训练新分类头（书 ch06 主线）
      - "last_transformer_stage"：冻结全部后，再解冻最后一个 transformer
        block（model.trf_blocks[-1]）与 final_norm + 新分类头（书 §6.5.2）
      - "all_layers"：基座全部可训练 + 新分类头
      - "lora"：冻结全部基座 → 新分类头保持全量可训（**不包 LoRA**）→
        其余 nn.Linear 加 LoRA 旁路（exclude_keywords=["out_head"]），
        只训 LoRA A/B + 分类头；不再解冻最后 block（书附录 E 用法）

    Args:
        model: models/gpt2 等 GPT 模型（需含 trf_blocks / final_norm / out_head）
        emb_dim: 嵌入维度（分类头输入维度）
        num_classes: 分类数
        finetune_method: 上述四种策略之一
        lora_rank / lora_alpha: LoRA 超参（缩放 = alpha/rank；
            rank=16, alpha=16 → 1 倍缩放，等价于无放大；想放大 16 倍可设
            alpha=256，见书附录 E 的参数选择讨论）

    Returns:
        改造后的 model（就地修改）
    """
    if finetune_method not in ("last_layer", "last_transformer_stage", "all_layers", "lora"):
        raise ValueError(
            f"未知 finetune_method: {finetune_method}，可选："
            "last_layer / last_transformer_stage / all_layers / lora"
        )

    # model architecture before finetune
    logger.info(f"model architecture before finetune: \n{model}")
    # model params numbers before freeze
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters before freeze: {total_params}")

    # 冻结全部基座参数
    for param in model.parameters():
        param.requires_grad = False

    # 统一替换输出层为新分类头（新建参数默认 requires_grad=True）
    model.out_head = nn.Linear(in_features=emb_dim, out_features=num_classes)

    if finetune_method == "last_transformer_stage":
        # 解冻最后一个 transformer block + 最终 LayerNorm（书 §6.5.2）
        blocks = getattr(model, "trf_blocks", None)
        if blocks is not None and len(blocks) > 0:
            for param in blocks[-1].parameters():
                param.requires_grad = True
        final_norm = getattr(model, "final_norm", None)
        if final_norm is not None:
            for param in final_norm.parameters():
                param.requires_grad = True
    elif finetune_method == "all_layers":
        # 全量微调：基座全部解冻
        for param in model.parameters():
            param.requires_grad = True
    elif finetune_method == "lora":
        # 只给 out_head 以外的 Linear 加 LoRA 旁路：
        # 基座 Linear 已冻结（requires_grad=False 被原样保留在包装内），
        # LoRA A/B 为新建参数默认可训练，out_head 保持全量可训且不被包裹
        replace_linear_with_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            exclude_keywords=["out_head"],
        )

    # model params numbers after finetune
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Total trainable parameters after finetune ({finetune_method}): {total_params}"
    )

    # model architecture after finetune
    logger.info(f"model architecture after finetune: \n{model}")

    return model




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
