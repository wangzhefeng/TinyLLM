# -*- coding: utf-8 -*-

# ***************************************************
# * File        : example_pretrained.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102020
# * Description : nano_vllm 加载真实 Qwen3 权重的示例。
# *               用法:
# *                 .venv/bin/python inference/serving/nano_vllm/example_pretrained.py \
# *                     --model-path ~/huggingface/Qwen3-0.6B/ "你好，请自我介绍"
# *               模型目录需为 HF 格式(含 config.json、*.safetensors、tokenizer 文件)，
# *               可用 inference/loaders/qwen3/model_download.py 下载。
# *               tokenizer 依赖 transformers；缺省时仅演示 token ids 输入输出。
# * Link        : https://github.com/GeeeekExplorer/nano-vllm
# * Requirement : torch >= 2.0, safetensors, transformers(可选, 用于文本 tokenizer)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
import argparse

# 双模式导入：包导入优先；直接以脚本运行时回退为插入仓根再包导入
try:
    from inference.serving.nano_vllm.config import Config
    from inference.serving.nano_vllm.sampling_params import SamplingParams
    from inference.serving.nano_vllm.engine.llm_engine import LLMEngine
    from inference.serving.nano_vllm.models.qwen3_for_causal_lm import Qwen3ForCausalLM
    from inference.serving.nano_vllm.utils.loader import load_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 仓根
    from inference.serving.nano_vllm.config import Config
    from inference.serving.nano_vllm.sampling_params import SamplingParams
    from inference.serving.nano_vllm.engine.llm_engine import LLMEngine
    from inference.serving.nano_vllm.models.qwen3_for_causal_lm import Qwen3ForCausalLM
    from inference.serving.nano_vllm.utils.loader import load_model

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 测试代码 main 函数
def main():
    parser = argparse.ArgumentParser(description="nano_vllm 真实权重推理示例")
    parser.add_argument("--model-path", type=str, required=True, help="HF 模型目录")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("prompt", type=str, nargs="?", default="hello")
    args = parser.parse_args()

    config = Config.from_pretrained(args.model_path, max_num_seqs=8)
    model = Qwen3ForCausalLM(config)
    load_model(model, args.model_path)
    print(f"权重加载完成: {args.model_path} "
          f"(layers={config.num_hidden_layers}, hidden={config.hidden_size}, vocab={config.vocab_size})")

    # tokenizer 可选：有 transformers 且目录含 tokenizer 文件则支持文本输入，否则要求纯 token ids
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        print(f"tokenizer 不可用({type(e).__name__})，prompt 将按空格分隔的 token id 解析")

    engine = LLMEngine(config, model=model, tokenizer=tokenizer, verbose=True)
    prompt = args.prompt if tokenizer is not None else [int(x) for x in args.prompt.split()]
    outputs = engine.generate(
        [prompt],
        SamplingParams(temperature=args.temperature, top_k=args.top_k, max_tokens=args.max_tokens),
    )
    print(f"\n生成 token_ids: {outputs[0]['token_ids']}")
    if outputs[0]["text"] is not None:
        print(f"生成文本: {outputs[0]['text']!r}")

if __name__ == "__main__":
    main()
