# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_gpt.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-15
# * Version     : 0.1.021500
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
import argparse

import torch

from pre_training.experiments.exp_pretrain_gpt2_124M import Model_Pretrain
from utils.args_tools import print_args_llm
from utils.device import torch_gc
from utils.random_seed import set_seed_torch
from utils.ddp_utils import (
    is_dist,
    init_dist, destory_dist,
    get_world_size, get_global_rank, get_local_rank,
    is_master, print_master_log,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def build_parser(description: str = "Model Pretraining") -> argparse.ArgumentParser:
    """
    预训练入口共享的参数解析器；legacy/其他模型入口可复用后 set_defaults 覆盖默认值。
    """
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description=description)
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--task_name", type=str,  default="model_pretrain", help="task name")
    parser.add_argument("--des", type=str,  default="tiny gpt pretrain", help="task description")
    parser.add_argument("--is_train", type=int,  default=1, help="train flag")
    parser.add_argument("--is_test", type=int,  default=0, help="test flag")
    parser.add_argument("--is_inference", type=int,  default=0, help="inference flag")
    # data params
    parser.add_argument("--data_source", type=str,  default="local", help="data source type")
    parser.add_argument("--data_url", type=str,  default=None, help="data source url")
    parser.add_argument("--data_path", type=str,  default="./dataset/pretraining", help="data file path")
    parser.add_argument("--data_file", type=str,  default="file.txt", help="data file")
    parser.add_argument("--data_name", type=str,  default="file", help="data file name")
    parser.add_argument("--train_ratio", type=float,  default=0.9, help="train data ratio")
    parser.add_argument("--batch_size", type=int,  default=8, help="batch size")
    parser.add_argument("--num_workers", type=int,  default=0, help="number of workers")
    # tokenizer params
    parser.add_argument("--tokenizer_model", type=str,  default="gpt2", help="tokenizer model")
    parser.add_argument("--vocab_size", type=int,  default=50257, help="vocab size")
    # model params
    parser.add_argument("--model_name", type=str,  default="gpt2_124M", help="model name")
    parser.add_argument("--context_length", type=int,  default=1024, help="context length")
    parser.add_argument("--embed_dim", type=int,  default=768, help="embedding dimension")
    parser.add_argument("--d_ff", type=int,  default=4*768, help="embedding dimension")
    parser.add_argument("--n_heads", type=int,  default=12, help="number of heads")
    parser.add_argument("--n_layers", type=int,  default=12, help="number of layers")
    parser.add_argument("--dropout", type=float,  default=0.1, help="dropout rate")
    parser.add_argument("--qkv_bias", action="store_false", default=True, help="use bias in qkv")
    parser.add_argument("--dtype", type=str,  default="bfloat16", help="dtype")
    # model pretrain params
    parser.add_argument("--use_amp", type=int,  default=1, help="Use amp")
    parser.add_argument("--learning_rate", type=float,  default=5e-4, help="learning rate")
    parser.add_argument("--initial_lr", type=float,  default=3e-5, help="initial learning rate")
    parser.add_argument("--min_lr", type=float,  default=1e-6, help="minimum learning rate")
    parser.add_argument("--weight_decay", type=float,  default=0.1, help="weight decay")
    parser.add_argument('--lradj', type = str, default = 'type1', help = 'adjust learning rate')
    # training schedule params (appendix D / ch05/10)
    parser.add_argument("--lr_schedule", type=str, default="warmup_cosine", choices=["warmup_cosine", "epoch"],
                        help="step-level warmup+cosine (book appendix D) or legacy epoch-level lradj")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="warmup steps ratio of total steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm, 0 to disable clipping")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="micro-batches per optimizer step")
    parser.add_argument("--pad_vocab_to_multiple_of", type=int, default=64,
                        help="pad vocab size up to a multiple (0/1 to disable); 50257 -> 50304")
    # checkpoint / resume params
    parser.add_argument("--save_every_steps", type=int, default=0,
                        help="save a resume checkpoint every N steps, 0 to disable")
    parser.add_argument("--keep_last_checkpoints", type=int, default=3, help="resume checkpoints to keep")
    parser.add_argument("--resume", type=str, default="",
                        help="resume from checkpoint: 'latest' or a checkpoint file path")
    parser.add_argument("--eos_token_id", type=int, default=50256, help="eos token id for sample generation")
    parser.add_argument("--seed", type=int,  default=42, help="seed")
    parser.add_argument("--itrs", type=int,  default=10, help="number of iterations")
    parser.add_argument("--train_epochs", type=int,  default=10, help="number of training epochs")
    parser.add_argument("--patience", type=int,  default=7, help="early stopping patience")
    parser.add_argument("--checkpoints", type=str, required=False, default="./saved_results/pretrained_models/", help="checkpoints path")
    parser.add_argument("--test_results", type=str, required=False, default="./saved_results/test_results/", help="test results path")
    # model inference params
    parser.add_argument("--max_new_tokens", type=int,  default=50, help="max new tokens")
    parser.add_argument("--use_cache", type=int,  default=0, help="max new tokens")
    # model pretrain device params
    parser.add_argument("--use_gpu", type=int,  default=1, help="user gpu")
    parser.add_argument("--gpu_type", type=str,  default="cuda", help="gpu type")
    parser.add_argument("--use_dp", type=int,  default=0, help="use dp")
    parser.add_argument("--use_ddp", type=int,  default=0, help="use ddp")
    # ------------------------------
    # arguments parse
    # ------------------------------
    return parser


def finalize_args(args):
    """
    parser 产物的公共后处理（dtype 字符串 -> torch.dtype）。
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]

    return args


def args_parse():
    args = finalize_args(build_parser().parse_args())

    return args


def run(args, global_rank, world_size, local_rank):
    # 模型任务
    Exp = Model_Pretrain
    # 模型训练
    if args.is_train:
        for itr in range(args.itrs):
            # setting record of experiments
            setting = f"{args.task_name}_{args.model_name}_dt{args.data_name}_cl{args.context_length}_te{args.train_epochs}_bs{args.batch_size}_itr{itr}"
            logger.info(f">>>>>>>training: iter-{itr} {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"{180 * '='}")
            # set experiments
            exp = Exp(args, global_rank, world_size, local_rank)
            # model training
            model = exp.train(
                training_iter=itr,
                setting=setting,
                eval_freq=5,
                eval_iter=1,
            )
            # model testing
            # if args.is_test:
            #     logger.info(f">>>>>>>testing: iter-{itr} {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            #     logger.info(f"{180 * '='}")
            #     exp.test(setting)

    # TODO 模型推理预测
    if args.is_inference:
        itr = 0
        setting = f"{args.task_name}_{args.model_name}_dt{args.data_name}_cl{args.context_length}_te{args.train_epochs}_bs{args.batch_size}_itr{itr}"
        logger.info(f">>>>>>>inference: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        prediction = exp.inference(setting, load=True)
        logger.info(prediction.shape)

    # empty cache
    logger.info(f"{180 * '='}")
    logger.info(f">>>>>>>>>>>> Empty cuda cache and memory pecices...")
    logger.info(f"{180 * '='}")
    torch_gc(device_id=local_rank)




# 测试代码 main 函数
def main(args, global_rank: int, world_size: int, local_rank: int):
    # Initialize process groups
    if is_dist():
        init_dist(global_rank, world_size)
    # 设置随机数
    set_seed_torch(seed = 2025)
    # Print info only on 1 GPU
    print_master_log(global_rank)
    # TODO All processes wait util rank 0 is done, using the GPU index.
    if is_dist():
        torch.distributed.barrier(device_ids=[local_rank])
    # 模型训练
    run(args, global_rank, world_size, local_rank)
    # Clean up distributed processes
    if is_dist():
        destory_dist()

if __name__ == "__main__":
    ###########################
    # 参数解析
    ###########################
    args = args_parse()
    print_args_llm(args)
    ###########################
    # 获取全局 rank（用于通信、日志、模型保存等）
    ###########################
    global_rank = get_global_rank()  # default: 0
    world_size = get_world_size()    # default: 1
    ###########################
    # 显式设置当前进程使用的 GPU 设备
    ###########################
    local_rank = get_local_rank()    # default: 0
    ###########################
    # Initiate training
    ###########################
    main(args, global_rank, world_size, local_rank)
    ###########################
    # TODO After training
    ###########################
    # Only create 1 plot
    if global_rank == 0:
        pass
