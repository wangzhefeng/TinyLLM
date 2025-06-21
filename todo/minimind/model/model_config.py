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
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from transformers import PretrainedConfig

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelConfig(PretrainedConfig):
    model_type = "llm"

    def __init__(self, 
                 dropout: float = 0.0,
                 bos_token_id: int = 1,
                 eos_token_id: int  = 2,
                 hidden_act: str = "silu",
                 hidden_size: int = 512,
                 intermediate_size: int = None,
                 max_position_embeddings: int = 32768,
                 num_attention_heads: int = 8,
                 num_hidden_layers: int = 8,
                 num_key_value_heads: int = 2,
                 vocab_size: int = 6400,
                 rms_norm_eps: float = 1e-05,
                 rope_theta: int = 1000000.0,
                 flash_attn: bool = True,
                 # specific config of MOE, when use_moe is false, the following is invalid
                 use_moe: bool = False,
                 num_experts_per_tok: int = 2,
                 n_routed_experts: int = 4,
                 n_shared_experts: int = 1,
                 scoring_func: str = "softmax",
                 aux_loss_alpha: float = 0.1,
                 seq_aux: bool = True,
                 norm_topk_prob: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        # specific config of MOE, when use_moe is False, the following is invalid
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个 token 选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为 'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的 alpha 参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化 top-k 概率




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
