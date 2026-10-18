# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_llama.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-01
# * Version     : 2.0.101826
# * Description : Llama2 预训练入口（LLMs-from-scratch ch05/14：用其他模型跑 ch05）。
# *               复用 run_gpt2_124M.py 的 parser 与训练主流程，默认 tiny 演示配置。
# *               真实 Llama2 分词器：--tokenizer_model llama2 --vocab_size 32000 --eos_token_id 2
# * Link        : https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/14_bonus_loop-over-other-LLMs
# * Requirement : torch
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from run_gpt2_124M import build_parser, finalize_args, main as pretrain_main
from utils.args_tools import print_args_llm
from utils.ddp_utils import get_global_rank, get_world_size, get_local_rank

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    parser = build_parser(description="Llama2 Pretraining")
    # tiny 演示配置（非 7B）；真实配置见 config/llama/model_cfgs.py
    parser.set_defaults(
        des="tiny llama2 pretrain",
        model_name="llama2",
        # 演示默认用 tiktoken GPT-2 分词器（无需下载 sentencepiece 模型），
        # vocab_size 与之保持一致；换 Llama2 分词器时改为 llama2/32000/2
        tokenizer_model="tiktoken_gpt2_bpe",
        vocab_size=50257,
        eos_token_id=50256,
        context_length=512,
        embed_dim=256,
        d_ff=1024,
        n_heads=8,
        n_layers=4,
        dropout=0.0,
        dtype="float32",
    )

    return finalize_args(parser.parse_args())


if __name__ == "__main__":
    # 参数解析
    args = args_parse()
    print_args_llm(args)
    # rank 信息（torchrun 或单进程）
    global_rank = get_global_rank()   # default: 0
    world_size = get_world_size()     # default: 1
    local_rank = get_local_rank()     # default: 0
    # 训练（与 run_gpt2_124M.py 同一主流程）
    pretrain_main(args, global_rank, world_size, local_rank)
