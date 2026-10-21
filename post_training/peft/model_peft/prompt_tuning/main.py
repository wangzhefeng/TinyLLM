# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-21
# * Version     : 2.0.102121
# * Description : Prompt Tuning 自研实现（soft prompt，不依赖 peft 库）。
# *               对照 Lester et al. 2021 与 LLMs-from-scratch 附录 E 的
# *               PEFT 主题：冻结基座 GPT，只训练 n 个可学习的虚拟 token
# *               embedding，拼接在真实 token embedding 之前。
# * Link        : Prompt Tuning https://arxiv.org/abs/2104.08691
# *               https://github.com/rasbt/LLMs-from-scratch （附录 E）
# * Requirement : torch（不引入新依赖）
# ***************************************************

__all__ = [
    "SoftPromptGPT",
    "freeze_base_model",
    "init_from_text",
    "count_trainable_params",
]

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
import os
os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
from utils.log_util import logger


def freeze_base_model(model: nn.Module):
    """冻结基座模型全部参数（prompt tuning 的前提：只训练 soft prompt）。"""
    for p in model.parameters():
        p.requires_grad = False


def count_trainable_params(model: nn.Module) -> int:
    """可训练参数量（教学上用于确认只有 soft prompt 在训练）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SoftPromptGPT(nn.Module):
    """带可学习 soft prompt 的 GPT 包装类（Lester et al. 2021）。

    结构：``num_virtual_tokens`` 个可学习 embedding 拼在真实 token embedding
    之前，其余前向与基座 GPT 完全一致（trf_blocks -> final_norm -> out_head）。

    位置编码约定：soft prompt 占用**前 n 个位置**（pos_embed[0:n]），真实
    token 的位置编码整体右移 n 位（pos_embed[n:n+seq]）——因果注意力下
    soft prompt 之间也可相互注意，是论文的标准放置方式。

    Args:
        base_model: models/gpt2/gpt2_124M.Model（需有 tok_embed/pos_embed/
            drop_embed/trf_blocks/final_norm/out_head 层名）
        num_virtual_tokens: soft prompt 长度 n（论文常用 10-100）
    """

    def __init__(self, base_model: nn.Module, num_virtual_tokens: int = 10):
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        embed_dim = base_model.tok_embed.embedding_dim
        # soft prompt：可学习的虚拟 token embedding（均值为 0 的小初始化）
        self.soft_prompt = nn.Embedding(num_virtual_tokens, embed_dim)
        nn.init.normal_(self.soft_prompt.weight, mean=0.0, std=0.02)
        # 冻结基座
        freeze_base_model(self.base_model)

    @property
    def cfg(self):
        return getattr(self.base_model, "cfg", None)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """前向：soft prompt + 真实 token，返回**只对应真实 token 位**的 logits。

        Args:
            x: (batch_size, seq_len) 真实 token ids（与基座 forward 同签名）
            use_cache: 兼容基座签名；soft prompt 场景通常不用 KV cache（教学实现不支持）

        Returns:
            (batch_size, seq_len, vocab_size) logits（不含虚拟 token 位）
        """
        base = self.base_model
        batch_size, seq_len = x.shape
        device = x.device
        n = self.num_virtual_tokens
        # 1) 真实 token embedding + 位置编码（位置整体右移 n：软 prompt 占前 n 位）
        tok_embeds = base.tok_embed(x)                          # (B, L, D)
        pos_id = torch.arange(n, n + seq_len, device=device)
        pos_embeds = base.pos_embed(pos_id).unsqueeze(0)        # (B, L, D)
        tok_x = tok_embeds + pos_embeds
        # 2) soft prompt embedding + 前 n 位位置编码，扩展到批维
        prompt_x = self.soft_prompt.weight.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_pos = base.pos_embed(torch.arange(n, device=device)).unsqueeze(0)
        prompt_x = prompt_x + prompt_pos                        # (B, n, D)
        # 3) 拼接并走完基座后半段
        x_in = torch.cat([prompt_x, tok_x], dim=1)              # (B, n+L, D)
        x_in = base.drop_embed(x_in)
        for block in base.trf_blocks:
            x_in = block(x_in, use_cache=False)
        x_in = base.final_norm(x_in)
        logits = base.out_head(x_in)                            # (B, n+L, V)
        # 只保留真实 token 位（虚拟位只用于提供条件信息，不产生损失）
        return logits[:, n:, :]

    def trainable_params(self) -> int:
        return count_trainable_params(self)


def init_from_text(soft_prompt_gpt: SoftPromptGPT, tokenizer, text: str,
                   fill: str = "mean"):
    """用真实词 embedding 初始化 soft prompt（论文的 token 初始化变体）。

    Args:
        tokenizer: 需有 encode(text) 方法（本仓 TiktokenBPE 等均可）
        text: 初始化用文本（如任务指令模板）
        fill: "mean" 用文本词 embedding 均值填所有虚拟位；
            "cyclic" 循环取文本各词 embedding 逐位填充（文本词数不足时循环）

    Returns:
        soft_prompt_gpt（就地初始化后返回）
    """
    token_ids = tokenizer.encode(text)
    if len(token_ids) == 0:
        raise ValueError("初始化文本不能为空")
    tok_weight = soft_prompt_gpt.base_model.tok_embed.weight.data
    init_vectors = tok_weight[token_ids]                          # (len, D)
    with torch.no_grad():
        if fill == "mean":
            mean_vec = init_vectors.mean(dim=0, keepdim=True)     # (1, D)
            soft_prompt_gpt.soft_prompt.weight.copy_(
                mean_vec.expand(soft_prompt_gpt.num_virtual_tokens, -1)
            )
        else:  # cyclic
            n = soft_prompt_gpt.num_virtual_tokens
            idx = [i % len(token_ids) for i in range(n)]
            soft_prompt_gpt.soft_prompt.weight.copy_(init_vectors[idx])
    return soft_prompt_gpt


# 测试代码 main 函数（玩具过拟合演示；规范测试见 tests/unit/test_prompt_tuning.py）
def main():
    from types import SimpleNamespace
    from models.gpt2.gpt2_124M import Model

    torch.manual_seed(42)
    cfg = SimpleNamespace(
        vocab_size=32, context_length=32, embed_dim=32, d_ff=64,
        n_heads=4, n_layers=2, dropout=0.0, qkv_bias=True,
        dtype=torch.float32, is_train=False,
    )
    base = Model(cfg)
    model = SoftPromptGPT(base, num_virtual_tokens=4)

    # 玩具任务：固定输入 -> 固定目标，验证只有 soft prompt 参与训练且 loss 下降
    x = torch.randint(0, 32, (1, 8))
    y = torch.tensor([[7] * 8])
    optimizer = torch.optim.Adam(model.soft_prompt.parameters(), lr=3e-2)
    first = last = None
    for _ in range(30):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    logger.info(f"toy overfit loss: {first:.3f} -> {last:.3f}")
    logger.info(f"trainable params: {model.trainable_params()} "
                f"(= num_virtual_tokens * embed_dim = 4 * 32)")


if __name__ == "__main__":
    main()
