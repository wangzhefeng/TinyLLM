# -*- coding: utf-8 -*-

# ***************************************************
# * File        : leaderboards.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-05
# * Version     : 2.0.102511
# * Description : Comparing models using preferences and leaderboards
# *               （reasoning-from-scratch chF/03）：两个区块——
# *               1) Elo 排行榜：elo_ratings 与教学辅助函数
# *                  expected_score / update_elo / format_leaderboard
# *                  （合并原 elo_leaderboard.py）；
# *               2) Bradley-Terry 排行榜：fit_bradley_terry 用
# *                  torch.optim.Adam 拟合 -logsigmoid(s_win - s_lose) 的
# *                  NLL，末位模型强度钉在 0（可辨识约束），再按
# *                  400/log(10) 缩放到 Elo 尺度并以 1000 为中心
# *                  （合并原 bradley_terry_leaderboard.py）。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/03_leaderboards
# *               Bradley-Terry https://jstor.org/stable/2333086
# * Requirement : torch（仅 fit_bradley_terry 需要）
# ***************************************************

__all__ = [
    "elo_ratings",
    "expected_score",
    "update_elo",
    "format_leaderboard",
    "fit_bradley_terry",
]

# python libraries
import math
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ==============================
# 区块 1：Elo 排行榜
# ==============================
def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):
    """
    Elo 评分系统: https://en.wikipedia.org/wiki/Elo_rating_system

    Args:
        vote_pairs (_type_): _description_
        k_factor (int, optional): _description_. Defaults to 32.
        initial_rating (int, optional): _description_. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    # Initialize all models with the same base rating
    ratings = {
        model: initial_rating
        for pair in vote_pairs
        for model in pair
    }
    # Update ratings after each match
    for winner, loser in vote_pairs:
        # Expected score for the current winner
        expected_winner = 1.0 / (1.0 + 10 ** ((ratings[loser] - ratings[winner]) / 400.0))
        # k_factor determines sensitivity of updates
        ratings[winner] = (ratings[winner] + k_factor * (1 - expected_winner))
        ratings[loser] = (ratings[loser] + k_factor * (0 - (1 - expected_winner)))

    return ratings


def expected_score(rating_a, rating_b):
    """Elo 期望胜率：``E_a = 1 / (1 + 10^((R_b - R_a) / 400))``。

    与 elo_ratings 内部使用的公式一致，单独拆出便于教学与单测；
    满足 ``expected_score(R_a, R_b) + expected_score(R_b, R_a) == 1``。
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_winner, rating_loser, k_factor=32):
    """单场对局后的 Elo 更新（胜者的实际得分 S=1，败者 S=0）。

    ``R' = R + K * (S - E)``；返回更新后的 (rating_winner, rating_loser)。
    平局（S=0.5）场景请按同式自行计算，此处只处理书 chF/03 的胜负对局。
    """
    expected_winner = expected_score(rating_winner, rating_loser)
    new_winner = rating_winner + k_factor * (1.0 - expected_winner)
    new_loser = rating_loser + k_factor * (0.0 - (1.0 - expected_winner))
    return new_winner, new_loser


def format_leaderboard(ratings):
    """把 {model: rating} 排成降序列表并格式化输出。"""
    lines = []
    for i, (model, score) in enumerate(sorted(ratings.items(), key=lambda x: -x[1]), 1):
        lines.append(f"{i:>2}. {model:<10} {score:7.1f}")
    return "\n".join(lines)


# ==============================
# 区块 2：Bradley-Terry 排行榜
# ==============================
def fit_bradley_terry(matches, iterations=400, lr=0.01, weight_decay=1e-4,
                      center=1000.0, seed=123):
    """Bradley-Terry 模型拟合（书 chF/03 的 2_bradley_terry_leaderboard.py 移植）。

    模型：``P(a 胜 b) = sigmoid(s_a - s_b)``，对每场对局最小化
    ``-logsigmoid(s_winner - s_loser)`` 的平均 NLL；n 个模型只学习
    n-1 个强度参数（最后一个钉 0，保证可辨识）。拟合后按
    ``scale = 400 / log(10)`` 把 logit 尺度强度换算成 Elo 尺度分数，
    再平移到以 ``center``（默认 1000）为中心。

    Args:
        matches: [(winner, loser), ...] 对局记录（模型名需可哈希）
        iterations: Adam 优化轮数
        lr: 学习率
        weight_decay: Adam 权重衰减（书原版为 1e-4）
        center: 缩放后的分数中心
        seed: 随机种子（保证可复现）

    Returns:
        list[(model, score)]：按分数降序排序的排行榜
    """
    torch.manual_seed(seed)

    # 收集全部模型并建立稳定索引（排序保证结果与输入顺序无关）
    models = sorted({m for winner, loser in matches for m in (winner, loser)})
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # 对局记录转为索引张量
    winners = torch.tensor([idx[winner] for winner, _ in matches], dtype=torch.long)
    losers = torch.tensor([idx[loser] for _, loser in matches], dtype=torch.long)

    # 可学习参数：n-1 个强度（末位恒 0）
    theta = torch.nn.Parameter(torch.zeros(n - 1))
    optimizer = torch.optim.Adam([theta], lr=lr, weight_decay=weight_decay)

    def scores():
        return torch.cat([theta, torch.zeros(1)])

    for _epoch in range(iterations):
        s = scores()
        delta = s[winners] - s[losers]                        # 强度差
        loss = -torch.nn.functional.logsigmoid(delta).mean()  # 负对数似然
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # logit 尺度 -> Elo 尺度（书原版：400/log(10)，中心 1000）
    with torch.no_grad():
        s = scores()
        R = s * (400.0 / math.log(10.0))
        R = R - R.mean() + center

    rating = {m: float(r) for m, r in zip(models, R.tolist())}
    leaderboard = sorted(rating.items(), key=lambda x: -x[1])
    return leaderboard



# 测试代码 main 函数
def main():
    votes = [
        ("GPT-5", "Claude-3"),
        ("GPT-5", "Llama-4"),
        ("Claude-3", "Llama-3"),
        ("Llama-4", "Llama-3"),
        ("Claude-3", "Llama-3"),
        ("GPT-5", "Llama-3"),
    ]

    ratings = elo_ratings(votes, k_factor=32, initial_rating=1000)

    print("Leaderboard (Elo)")
    print("-----------------------------")
    for model in sorted(ratings, key=ratings.get, reverse=True):
        print(f"{model:8s} : {ratings[model]:.1f}")

    # Bradley-Terry 排行榜（mock 对局数据拟合演示）
    print("\nLeaderboard (Bradley-Terry)")
    print("-----------------------------")
    leaderboard = fit_bradley_terry(votes, iterations=400, lr=0.01)
    for i, (model, score) in enumerate(leaderboard, 1):
        print(f"{i:>2}. {model:<10} {score:7.1f}")
    print()


if __name__ == "__main__":
    main()
