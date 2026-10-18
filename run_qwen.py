# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_qwen.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-17
# * Version     : 2.0.101826
# * Description : Qwen3 预训练入口（LLMs-from-scratch ch05/14：用其他模型跑 ch05）。
# *               复用 run_gpt2_124M.py 的 parser 与训练主流程，默认 tiny 演示配置
# *               （dense，num_experts=0）；真实配置见 config/qwen3/model_cfgs.py。
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
    parser = build_parser(description="Qwen3 Pretraining")
    # Qwen3 结构参数（GQA / RoPE / 可选 MoE）
    parser.add_argument("--n_kv_groups", type=int, default=2, help="GQA key-value groups")
    parser.add_argument("--head_dim", type=int, default=0, help="attention head dim, 0 = embed_dim // n_heads")
    parser.add_argument("--qk_norm", type=int, default=1, help="normalize queries/keys in GQA")
    parser.add_argument("--rope_base", type=float, default=1_000_000.0, help="RoPE theta base")
    parser.add_argument("--num_experts", type=int, default=0, help="MoE experts, 0 = dense FFN")
    # tiny 演示配置（非 0.6B）
    parser.set_defaults(
        des="tiny qwen3 pretrain",
        model_name="qwen3",
        # 演示默认用 tiktoken GPT-2 分词器（无需下载 Qwen3 分词文件），
        # vocab_size 与之保持一致；换 Qwen3 分词器时 vocab_size 用 151936、eos 151643
        tokenizer_model="tiktoken_gpt2_bpe",
        vocab_size=50257,
        eos_token_id=50256,
        context_length=512,
        embed_dim=256,
        d_ff=1024,
        n_heads=8,
        n_layers=4,
        dropout=0.0,
        # Qwen3 block 内部固定以 bfloat16 计算（layers/transformer_block.py），dtype 必须是 bf16
        dtype="bfloat16",
    )
    args = parser.parse_args()
    # head_dim = 0 表示未指定，交给模型按 embed_dim // n_heads 计算
    if args.head_dim == 0:
        args.head_dim = None
    args.qk_norm = bool(args.qk_norm)

    return finalize_args(args)


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
