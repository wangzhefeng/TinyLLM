# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_gpt_pretrain.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-15
# * Version     : 0.1.021500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
from pathlib import Path

from exp.exp_pretrain_gpt import Model_Pretrain
from utils.args_tools import print_args
from utils.device import torch_gc
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description="Model Pretraining")
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--task_name", type=str, required=True, default="model_pretrain", help="task name")
    parser.add_argument("--is_train", type=int, required=True, default=1, help="train flag")
    parser.add_argument("--is_test", type=int, required=True, default=0, help="test flag")
    parser.add_argument("--is_inference", type=int, required=True, default=0, help="inference flag")
    # data params
    parser.add_argument("--data_path", type=str, required=True, default="./dataset/pretrain/gpt/file.txt", help="data download url")
    parser.add_argument("--context_length", type=int, required=True, default=1024, help="context length")
    # model params
    parser.add_argument("--model_name", type=str, required=True, default="gpt", help="model name")
    parser.add_argument("--vocab_size", type=int, required=True, default=50257, help="vocab size")
    parser.add_argument("--emb_dim", type=int, required=True, default=768, help="embedding dimension")
    parser.add_argument("--n_heads", type=int, required=True, default=12, help="number of heads")
    parser.add_argument("--n_layers", type=int, required=True, default=12, help="number of layers")
    parser.add_argument("--dropout", type=float, required=True, default=0.1, help="dropout rate")
    parser.add_argument("--qkv_bias", type=int, required=True, default=0, help="use bias in qkv")
    parser.add_argument("--dtype", type=str, required=True, default="float16", help="dtype")
    parser.add_argument("--max_new_tokens", type=int, required=True, default=50, help="max new tokens")
    # model pretrain params
    parser.add_argument("--iters", type=int, required=True, default=10, help="number of iterations")
    parser.add_argument("--train_epochs", type=int, required=True, default=10, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True, default=2, help="batch size")
    parser.add_argument("--train_ratio", type=float, required=True, default=0.9, help="train data ratio")
    parser.add_argument("--learning_rate", type=float, required=True, default=5e-4, help="learning rate")
    parser.add_argument("--initial_lr", type=float, required=True, default=3e-5, help="initial learning rate")
    parser.add_argument("--min_lr", type=float, required=True, default=1e-6, help="minimum learning rate")
    parser.add_argument("--weight_decay", type=float, required=True, default=0.1, help="weight decay")
    parser.add_argument('--lradj', type = str, default = 'type1', help = 'adjust learning rate')
    parser.add_argument("--patience", type=int, required=True, default=7, help="early stopping patience")
    parser.add_argument("--checkpoints", type=str, required=False, default="./saved_results/pretrained_models/", help="checkpoints path")
    parser.add_argument("--test_results", type=str, required=False, default="./saved_results/test_results/", help="test results path")
    parser.add_argument("--use_amp", type=int, required=True, default=1, help="Use amp")
    # model pretrain device params
    parser.add_argument("--use_gpu", type=int, required=True, default=1, help="user gpu") 
    parser.add_argument("--gpu_type", type=str, required=True, default="cuda", help="gpu type")
    parser.add_argument("--use_multi_gpu", type=int, required=True, default=0, help="use multi gpu")
    parser.add_argument("--devices", type=str, required=True, default="0,1,2,3,4,5,6,7", help="devices") 
    # ------------------------------
    # arguments parse
    # ------------------------------
    args = parser.parse_args()

    return args


def run(args):
    # 模型任务
    if args.task_name == 'tiny_gpt_pretrain':
        Exp = Model_Pretrain
    else:
        Exp = Model_Pretrain
    # 模型训练
    if args.is_train:
        for itr in range(args.iters):
            # setting record of experiments
            setting = f"{args.task_name}_{args.model_name}_{args.data[:-4]}_cl{args.context_length}_te{args.train_epochs}_bs{args.batch_size}"
            logger.info(f">>>>>>>training: iter-{itr} {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"{180 * '='}")
            # set experiments
            exp = Exp(args)
            # model training
            exp.train(training_iter = itr, setting=setting, eval_freq=5, eval_iter=1, start_context="Every effort moves you")
            # model testing
            if args.is_test:
                logger.info(f">>>>>>>testing: iter-{itr} {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                logger.info(f"{180 * '='}")
                exp.test(setting)
    
    # 模型推理预测
    if args.is_inference:
        itr = 0
        logger.info(f">>>>>>>inference: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        prediction = exp.predict(setting, True)
        logger.info(prediction.shape)

    # empty cache
    logger.info(f"{180 * '='}")
    logger.info(f">>>>>>>>>>>> Empty cuda cache and memory pecices...")
    logger.info(f"{180 * '='}")
    torch_gc(gpu_type=args.gpu_type, device=exp.device)




# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed(seed = 2025)
    # 参数解析
    args = args_parse() 
    # TODO print_args(args)
    logger.info(f"Args in experiment: \n{args}")
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
