# -*- coding: utf-8 -*-

# ***************************************************
# * File        : example.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-28
# * Version     : 1.0.072816
# * Description : nano_vllm 教学示例(纯 PyTorch / CPU / 单进程)。
# *               使用随机初始化的微型 Qwen3(hidden=128, 2 层, 4 头, vocab=68)
# *               + 伪字符 tokenizer，演示：
# *               1. continuous batching: 多条请求在同一步中混合 prefill/decode
# *               2. 分页 KV cache: block_size=16 的 block 分配与复用
# *               3. chunked prefill: max_num_batched_tokens 小于 prompt 总长时分块
# *               运行: cd TinyLLM && .venv/bin/python inference/serving/nano_vllm/example.py
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
PKG_ROOT = str(Path(__file__).resolve().parent)
if PKG_ROOT in sys.path:
    sys.path.remove(PKG_ROOT)
sys.path.insert(0, PKG_ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch

from config import Config
from sampling_params import SamplingParams
from engine.llm_engine import LLMEngine

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class CharTokenizer:
    """
    伪 tokenizer：字符 <-> id 的一一映射(0 留给 <eos>)
    """

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyz .,!?")
        self.itos = ["<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text.lower() if c in self.stoi]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[i] for i in token_ids if 0 <= i < self.vocab_size)




# 测试代码 main 函数
def main():
    torch.manual_seed(42)
    tokenizer = CharTokenizer()

    # 随机初始化的微型 Qwen3 配置(hidden=128, 2 层, 4 头, 2 KV 头)
    config = Config(
        # 模型结构
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        # 引擎调度(故意调小以展示 chunked prefill / continuous batching)
        max_num_batched_tokens=24,   # 小于最长 prompt 长度 -> 触发分块 prefill
        max_num_seqs=4,              # 单步最多 4 条序列
        max_model_len=96,
        eos=tokenizer.eos_token_id,
        kvcache_block_size=16,       # 分页 KV cache 的 block 大小
        num_kvcache_blocks=64,       # 共 64 * 16 = 1024 个 token 槽位
    )
    print("=" * 72)
    print("nano_vllm 教学示例：微型随机 Qwen3 + continuous batching + paged KV cache")
    print("=" * 72)
    print(f"模型配置: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"vocab={config.vocab_size}")
    print(f"引擎配置: max_num_batched_tokens={config.max_num_batched_tokens}, "
          f"max_num_seqs={config.max_num_seqs}, block_size={config.kvcache_block_size}, "
          f"num_blocks={config.num_kvcache_blocks}")

    engine = LLMEngine(config, tokenizer=tokenizer, verbose=True)

    prompts = [
        "hello tiny world. this is a long prompt for chunked prefill demo!",
        "cat sat on mat",
        "what is paged attention?",
        "continuous batching demo",
    ]
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=8),                 # 贪心
        SamplingParams(temperature=0.8, top_k=5, max_tokens=12),       # 温度 + top_k
        SamplingParams(temperature=1.0, top_k=10, max_tokens=10),
        SamplingParams(temperature=0.6, top_k=3, max_tokens=14),
    ]
    print("\n--- 调度过程 (verbose) ---")
    outputs = engine.generate(prompts, sampling_params)

    print("\n--- 生成结果 ---")
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"[请求 {i}] prompt({len(tokenizer.encode(prompt))} tokens): {prompt!r}")
        print(f"          生成 token_ids: {output['token_ids']}")
        print(f"          解码文本: {output['text']!r}")
    print("\n示例运行完成：随机权重模型的输出无语义，但 prefill -> decode 闭环、"
          "continuous batching、分页 KV cache 分配/释放均已真实执行。")

if __name__ == "__main__":
    main()
