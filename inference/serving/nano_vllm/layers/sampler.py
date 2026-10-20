# -*- coding: utf-8 -*-

# ***************************************************
# * File        : sampler.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : 采样器：temperature + top_k 采样。
# *               上游仅支持 temperature(Gumbel-max 技巧)，本实现额外支持
# *               逐序列 top_k 截断，temperature=0 时退化为贪心(argmax)。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Sampler",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Sampler(nn.Module):
    """
    采样器

    - temperature > 0: 温度缩放 + (可选)top_k 截断 + multinomial 采样
    - temperature = 0: 贪心 argmax
    """

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [bs, vocab_size]
            temperatures: [bs]，0 表示贪心
            top_ks: [bs]，-1 表示不截断
        Returns:
            sample_tokens: [bs]
        """
        logits = logits.float()
        greedy = temperatures <= 0
        # 温度缩放(贪心行温度置 1，避免除零)
        safe_temps = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        logits = logits / safe_temps.unsqueeze(dim=1)
        # top_k 截断：按行屏蔽不在前 k 名的 token
        if top_ks is not None:
            ks = top_ks.tolist()
            for i, k in enumerate(ks):
                if 0 < k < logits.size(-1):
                    threshold = logits[i].topk(k).values[-1]
                    logits[i] = logits[i].masked_fill(logits[i] < threshold, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
        if greedy.all():
            sample_tokens = probs.argmax(dim=-1)
        else:
            # 贪心行直接 argmax，其余行 multinomial
            sample_tokens[greedy] = probs[greedy].argmax(dim=-1)
            rand_rows = ~greedy
            if rand_rows.any():
                sample_tokens[rand_rows] = torch.multinomial(probs[rand_rows], num_samples=1).squeeze(-1)
        return sample_tokens




# 测试代码 main 函数
def main():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.randn(4, 100)
    temperatures = torch.tensor([0.0, 1.0, 1.0, 0.0])
    top_ks = torch.tensor([-1, 5, 10, -1])
    tokens = sampler(logits, temperatures, top_ks)
    assert tokens.shape == (4,)
    # 贪心行应等于 argmax
    assert tokens[0].item() == logits[0].argmax().item()
    assert tokens[3].item() == logits[3].argmax().item()
    print("Sampler 测试通过:", tokens.tolist())

if __name__ == "__main__":
    main()
