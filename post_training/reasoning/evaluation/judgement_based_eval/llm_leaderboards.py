# -*- coding: utf-8 -*-

# ***************************************************
# * File        : leaderboards.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-05
# * Version     : 1.0.100516
# * Description : Comparing models using preferences and leaderboards
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


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
    print(ratings)
    # Update ratings after each match
    for winner, loser in vote_pairs:
        # Expected score for the current winner
        expected_winner = 1.0 / (1.0 + 10 ** ((ratings[loser] - ratings[winner]) / 400.0))
        # k_factor determines sensitivity of updates
        ratings[winner] = (ratings[winner] + k_factor * (1 - expected_winner))
        ratings[loser] = (ratings[loser] + k_factor * (0 - (1 - expected_winner)))

    return ratings




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
    print(ratings)
    
    for model in sorted(ratings, key=ratings.get, reverse=True):
        print(f"{model:8s} : {ratings[model]:.1f}")

if __name__ == "__main__":
    main()
