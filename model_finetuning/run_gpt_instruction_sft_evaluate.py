# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_gpt_instruction_sft_evaluate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-15
# * Version     : 1.0.031520
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse

import torch

from exp.exp_finetune_gpt_instruction_flow_evaluate import (
    ModelFinetuningInstructionFlowEvaluate
)
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def args_parse():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description="Tiny GPT Finetuning Instruction Flow")
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--task_name", type=str, required=True, 
                        default="tiny_gpt_instruction_sft_evaluate",
                        help="task name")
    parser.add_argument("--model_name", type=str, required=True, 
                        default="gpt_instruction_sft",
                        help="model name") 
    # data params
    parser.add_argument("--data_source", type=str, required=True, 
                        default="./dataset/finetune/instruction-data-with-response.json", 
                        help="data path")
    parser.add_argument("--train_ratio", type=float, required=True, 
                        default=0.85,
                        help="train data ratio")
    parser.add_argument("--test_ratio", type=float, required=True, 
                        default=0.10,
                        help="test data ratio") 
    # model params
    parser.add_argument("--inference_server", type=str, required=True, 
                        default="ollama",
                        help="inference server")
    parser.add_argument("--inference_server_url", type=str, required=True, 
                        default="http://localhost:11434/api/chat",
                        help="inference server url")
    parser.add_argument("--inference_model", type=str, required=True, 
                        default="llama3",
                        help="inference model")
    parser.add_argument("--num_ctx", type=int, required=True,
                        default=2048,
                        help="number of context")
    parser.add_argument("--iters", type=int, required=True, default=10, 
                        help="number of iterations")
    parser.add_argument("--seed", type=int, required=True,
                        default=123,
                        help="random seed")
    # model pretrain device params
    parser.add_argument("--num_workers", type=int, required=True, 
                        default=0,
                        help="num_workers")
    parser.add_argument("--use_gpu", type=int, required=True, 
                        default=1, 
                        help="user gpu")
    parser.add_argument("--use_multi_gpu", type=int, required=True, 
                        default=0, 
                        help="use multi gpu")
    parser.add_argument("--gpu_type", type=str, required=True, 
                        default="cuda", 
                        help="gpu type")
    parser.add_argument("--devices", type=str, required=True, 
                        default="0,1,2,3",
                        help="devices")
    # ------------------------------
    # arguments parse
    # ------------------------------
    args = parser.parse_args()
    # use gpu
    args.use_gpu = True \
        if (torch.cuda.is_available() or torch.backends.mps.is_available()) and args.use_gpu \
        else False
    # gpu type: "cuda", "mps"
    args.gpu_type = args.gpu_type.lower().strip()
    # devices string: "0,1,2,3", "0", "1", "2", "3", "0,1", "0,2"...
    args.devices = args.devices.replace(" ", "")
    # device ids: [0,1,2,3], [0], [1], [2], [3], [0,1], [0,2]...
    args.device_ids = [int(id_) for id_ in args.devices.split(",")]
    # gpu: [0,1,2,3], "0"
    if args.use_gpu and args.use_multi_gpu:
        args.gpu = args.devices
    elif args.use_gpu and not args.use_multi_gpu:
        args.gpu = args.device_ids[0]
    
    logger.info(f"Args in experiment: \n{args}")

    return args


def run(args):
    # ------------------------------
    # 模型任务
    # ------------------------------
    if args.task_name == 'tiny_gpt_instruction_sft_evaluate':
        Exp = ModelFinetuningInstructionFlowEvaluate
    else:
        Exp = ModelFinetuningInstructionFlowEvaluate
    
    # setting record of experiments
    setting = f"{args.task_name}_{args.model_name}_{args.data_source.split('/')[-1][:-6]}"
    # ------------------------------
    # 模型训练
    # ------------------------------
    for itr in range(args.iters):
        logger.info(f"{50 * '='}")
        logger.info(f"model evaluate iter: {itr}")
        logger.info(f"{50 * '='}")

        # set experiments
        exp = Exp(args)

        # model evaluate
        exp.evaluate()




# 测试代码 main 函数
def main():
    # 参数解析
    args = args_parse()
    # 设置随机数
    set_seed(args.seed)
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
