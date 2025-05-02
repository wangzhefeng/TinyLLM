# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Config.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-09
# * Version     : 0.1.020920
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import List

from transformers import PretrainedConfig

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ModelConfig(PretrainedConfig):
    model_type = "llm"

    def __init__(self, 
                 dim: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 16,
                 n_kv_heads: int = 8,
                 vocab_size: int = 6400,
                 hidden_dim: int = None,
                 multiple_of: int = 64,
                 norm_eps: float = 1e-5,
                 max_seq_len: int = 512,
                 dropout: float = 0.0,
                 flash_attn: bool = True, 
                 use_moe: bool = False,
                 num_experts_per_tok: int = 2,
                 n_routed_experts: int = 4,
                 n_shared_experts: bool = True,
                 scoring_func: str = "softmax",
                 aux_loss_alpha: float = 0.01,
                 seq_aux: bool = True,
                 norm_topk_prob: bool = True,
                 **kwargs):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        # specific config of MOE, when use_moe is False, 
        # the following is invalid
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super(ModelConfig, self).__init__(**kwargs)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
