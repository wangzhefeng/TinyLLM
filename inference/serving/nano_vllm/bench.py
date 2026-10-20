# -*- coding: utf-8 -*-

# ***************************************************
# * File        : bench.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : nano_vllm 吞吐基准(移植自上游 bench.py)。
# *               默认用随机初始化的微型 Qwen3(CPU 可跑，验证引擎吞吐)；
# *               传入 --model-path 指向 HF 格式目录(如 ~/huggingface/Qwen3-0.6B/)
# *               则加载真实权重基准。与上游的区别：单进程 CPU，
# *               数值不可与上游 GPU 结果直接对比，仅用于回归与教学。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0, safetensors(仅 --model-path 模式)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
import argparse
import time
from random import randint, seed

import torch

# 双模式导入：包导入优先；直接以脚本运行时回退为插入仓根再包导入
try:
    from inference.serving.nano_vllm.config import Config
    from inference.serving.nano_vllm.sampling_params import SamplingParams
    from inference.serving.nano_vllm.engine.llm_engine import LLMEngine
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓根
    from inference.serving.nano_vllm.config import Config
    from inference.serving.nano_vllm.sampling_params import SamplingParams
    from inference.serving.nano_vllm.engine.llm_engine import LLMEngine

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def build_engine(args):
    """构造引擎：--model-path 加载真实权重，否则随机初始化微型模型"""
    if args.model_path:
        from inference.serving.nano_vllm.models.qwen3_for_causal_lm import Qwen3ForCausalLM
        from inference.serving.nano_vllm.utils.loader import load_model
        config = Config.from_pretrained(
            args.model_path,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.num_seqs,
            max_model_len=args.max_model_len,
            kvcache_block_size=args.block_size,
        )
        model = Qwen3ForCausalLM(config)
        load_model(model, args.model_path)
        print(f"加载真实权重: {args.model_path} "
              f"(layers={config.num_hidden_layers}, hidden={config.hidden_size})")
    else:
        from inference.serving.nano_vllm.models.qwen3_for_causal_lm import Qwen3ForCausalLM
        from inference.serving.nano_vllm.utils.loader import init_random_weights
        config = Config(
            vocab_size=32000, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=max(512, args.max_model_len),
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.num_seqs, max_model_len=args.max_model_len,
            eos=0, kvcache_block_size=args.block_size,
        )
        model = init_random_weights(Qwen3ForCausalLM(config))
        print("使用随机初始化的微型 Qwen3(hidden=128, 2 层)")
    return LLMEngine(config, model=model), config


# 测试代码 main 函数
def main():
    parser = argparse.ArgumentParser(description="nano_vllm 吞吐基准")
    parser.add_argument("--model-path", type=str, default=None,
                        help="HF 模型目录(含 config.json 与 *.safetensors)，缺省用随机微型模型")
    parser.add_argument("--num-seqs", type=int, default=64, help="并发请求数(上游为 256)")
    parser.add_argument("--max-input-len", type=int, default=128, help="单条 prompt 最大长度(上游为 1024)")
    parser.add_argument("--max-output-len", type=int, default=128, help="单条生成最大长度(上游为 1024)")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    seed(0)
    torch.manual_seed(0)
    engine, config = build_engine(args)

    prompt_token_ids = [
        [randint(1, config.vocab_size - 1) for _ in range(randint(16, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, top_k=20, ignore_eos=True,
                       max_tokens=randint(16, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    # warmup
    engine.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=4))
    t = time.time()
    engine.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {total_tokens / t:.2f}tok/s")

if __name__ == "__main__":
    main()
