# -*- coding: utf-8 -*-

# ***************************************************
# * File        : __init__.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102421
# * Description : Reward Model 训练（RLHF 一阶段）。
# * Requirement : torch
# ***************************************************

__all__ = ["RewardModel", "bradley_terry_loss", "rm_pairwise_step"]

from post_training.preference.rm.rm_head import (  # noqa: F401
    RewardModel,
    bradley_terry_loss,
    rm_pairwise_step,
)
