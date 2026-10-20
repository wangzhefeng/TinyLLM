# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : nano_vllm 引擎 + 模型配置。
# *               与上游 nanovllm/config.py 的区别：
# *               1. 不依赖 transformers.AutoConfig，模型结构参数直接内嵌；
# *               2. 纯 CPU 运行，去除 gpu_memory_utilization 等 CUDA 参数；
# *               3. kvcache_block_size 不再强制 256 对齐（教学用小 block）。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = [
    "Config",
]

# python libraries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field

import torch

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


@dataclass
class Config:
    """
    nano_vllm 全局配置（模型结构参数 + 引擎调度参数）

    Args:
        模型结构参数(对应 HF Qwen3Config):
            vocab_size: 词表大小
            hidden_size: 隐藏层维度
            intermediate_size: MLP 中间层维度
            num_hidden_layers: decoder 层数
            num_attention_heads: 注意力头数
            num_key_value_heads: KV 头数(GQA)，等于 num_attention_heads 时为 MHA
            head_dim: 每个注意力头的维度，默认 hidden_size // num_attention_heads
            max_position_embeddings: 最大位置编码长度
            rms_norm_eps: RMSNorm 的 eps
            rope_theta: RoPE 的 base
            attention_bias: qkv_proj 是否带 bias(Qwen3 默认 False)
            tie_word_embeddings: lm_head 与 embed_tokens 是否共享权重
            hidden_act: MLP 激活函数(仅支持 "silu")
        引擎调度参数:
            max_num_batched_tokens: 单次 step 最多调度的 token 数(prefill 分块依据)
            max_num_seqs: 单次 step 最多调度的序列数(continuous batching 上限)
            max_model_len: 单条序列最大长度(prompt + completion)
            eos: eos token id，命中后序列结束(除非 ignore_eos)
            kvcache_block_size: paged KV cache 每个 block 容纳的 token 数
            num_kvcache_blocks: KV cache 总 block 数，-1 表示按 max_num_seqs/max_model_len 估算
            tensor_parallel_size: 张量并行 world_size，>1 时由 LLMEngine spawn 子进程(gloo 后端)
            enforce_eager: 是否禁用 CUDA graph(True 则始终 eager；CPU 上无 CUDA graph 概念)
            tp_master_port: gloo init 的 tcp 端口，-1 表示由 LLMEngine 动态选择空闲端口
            dtype: 计算精度，CPU 上使用 float32
    """
    # 模型结构参数
    vocab_size: int = 32000
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = -1
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    # 引擎调度参数
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    eos: int = -1
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    tp_master_port: int = -1
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        assert 1 <= self.tensor_parallel_size <= 8, "张量并行 world_size 需在 [1, 8]"
        assert self.hidden_act == "silu"
        assert self.kvcache_block_size >= 1
        if self.head_dim == -1:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        # TP 可分片性检查
        tp = self.tensor_parallel_size
        assert self.num_attention_heads % tp == 0, "注意力头数需可被 tensor_parallel_size 整除"
        assert self.num_key_value_heads % tp == 0, "KV 头数需可被 tensor_parallel_size 整除"
        assert self.vocab_size % tp == 0, "词表大小需可被 tensor_parallel_size 整除"
        assert self.intermediate_size % tp == 0, "intermediate_size 需可被 tensor_parallel_size 整除"
        self.max_model_len = min(self.max_model_len, self.max_position_embeddings)
        # 未显式指定 KV cache block 数时，按 "所有并发序列都跑满 max_model_len" 估算
        if self.num_kvcache_blocks == -1:
            blocks_per_seq = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
            self.num_kvcache_blocks = max(4, self.max_num_seqs * blocks_per_seq)

    @classmethod
    def from_pretrained(cls, path: str, **overrides) -> "Config":
        """
        从 HF 模型目录读取 config.json(和 generation_config.json 的 eos)构造 Config。
        引擎调度参数可用关键字覆盖，如 Config.from_pretrained(path, max_num_seqs=64)。

        Args:
            path: HF 模型目录(含 config.json 与 *.safetensors)
        """
        import json
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            hf = json.load(f)
        fields = {
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "rms_norm_eps", "rope_theta",
            "attention_bias", "tie_word_embeddings", "hidden_act",
        }
        kwargs = {k: v for k, v in hf.items() if k in fields}
        # eos 优先取 generation_config.json，其次 config.json
        gen_cfg_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_cfg_path):
            with open(gen_cfg_path, encoding="utf-8") as f:
                gen_cfg = json.load(f)
            eos = gen_cfg.get("eos_token_id")
            kwargs["eos"] = eos[-1] if isinstance(eos, list) else eos
        elif hf.get("eos_token_id") is not None:
            kwargs["eos"] = hf["eos_token_id"]
        kwargs.update(overrides)
        return cls(**kwargs)




# 测试代码 main 函数
def main():
    config = Config(vocab_size=64, max_model_len=128, max_num_seqs=4)
    print(config)

if __name__ == "__main__":
    main()
