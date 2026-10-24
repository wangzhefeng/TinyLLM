# -*- coding: utf-8 -*-

# ***************************************************
# * File        : elo_leaderboard.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-12
# * Version     : 2.0.102101
# * Description : Elo 排行榜（reasoning-from-scratch chF/03）。
# *               在线 Elo 更新的规范实现位于 llm_leaderboards.py
# *               （elo_ratings），本文件 re-export 之并补充两个教学辅助
# *               函数（expected_score / update_elo），拆开 Elo 更新公式
# *               的两步，便于单测与讲解；不再重复维护第二份实现。
# * Link        : https://github.com/rasbt/reasoning-from-scratch/tree/main/chF/03_leaderboards
# *               https://en.wikipedia.org/wiki/Elo_rating_system
# * Requirement : 无（纯 Python）
# ***************************************************

__all__ = [
    "elo_ratings",
    "expected_score",
    "update_elo",
    "format_leaderboard",
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

# 规范实现 re-export（消除与 llm_leaderboards.py 的重复）
from inference.evaluation.judges.llm_leaderboards import (
    elo_ratings,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


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



# 测试代码 main 函数
def main():
    """离线演示：mock 对局数据计算 Elo 排行榜与教学辅助函数。"""
    votes = [
        ("GPT-5", "Claude-3"),
        ("GPT-5", "Llama-4"),
        ("Claude-3", "Llama-3"),
        ("Llama-4", "Llama-3"),
        ("Claude-3", "Llama-3"),
        ("GPT-5", "Llama-3"),
    ]

    ratings = elo_ratings(votes, k_factor=32, initial_rating=1000)
    logger.info(f"\nLeaderboard (Elo)\n-----------------\n{format_leaderboard(ratings)}")

    # 教学辅助：期望胜率与单场更新
    logger.info(
        f"expected_score(1000, 1000) = {expected_score(1000, 1000)}（同级对半）"
    )
    logger.info(
        f"update_elo(1000, 1000) -> {update_elo(1000, 1000)}（胜者+16/败者-16）"
    )


if __name__ == "__main__":
    main()
