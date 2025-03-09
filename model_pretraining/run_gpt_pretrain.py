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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import argparse

import torch

from exp.exp_pretrain_gpt import Model_Pretrain
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def args_parse():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description="Tiny GPT Pretraining")
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--task_name", type=str, required=True, default="tiny_gpt_pretrain",
                        help="task name")
    parser.add_argument("--is_train", type=int, required=True, default=1,
                        help="training flag")
    parser.add_argument("--is_inference", type=int, required=True, default=0,
                        help="inference flag")
    # data params
    parser.add_argument("--data_source", type=str, required=True, 
                        default="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt", 
                        help="data download url")
    parser.add_argument("--context_length", type=int, required=True, default=1024,
                        help="context length")
    # model params
    parser.add_argument("--model_name", type=str, required=True, default="gpt",
                        help="model name")
    parser.add_argument("--vocab_size", type=int, required=True, default=50257,
                        help="vocab size")
    parser.add_argument("--emb_dim", type=int, required=True, default=768,
                        help="embedding dimension")
    parser.add_argument("--n_heads", type=int, required=True, default=12,
                        help="number of heads")
    parser.add_argument("--n_layers", type=int, required=True, default=12,  
                        help="number of layers")
    parser.add_argument("--dropout", type=float, required=True, default=0.1, 
                        help="dropout")
    parser.add_argument("--qkv_bias", type=int, required=True, default=0, 
                        help="use bias in qkv")
    parser.add_argument("--max_new_tokens", type=int, required=True, default=50,
                        help="max new tokens")
    # model pretrain params
    parser.add_argument("--iters", type=int, required=True, default=10, 
                        help="number of iterations")
    parser.add_argument("--train_epochs", type=int, required=True, default=10, 
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True, default=2, 
                        help="batch size")
    parser.add_argument("--train_ratio", type=float, required=True, default=0.9, 
                        help="train data ratio")
    parser.add_argument("--learning_rate", type=float, required=True, default=5e-4, 
                        help="learning rate")
    parser.add_argument("--initial_lr", type=float, required=True, default=3e-5, 
                        help="initial learning rate")
    parser.add_argument("--min_lr", type=float, required=True, default=1e-6,
                        help="minimum learning rate")
    parser.add_argument("--weight_decay", type=float, required=True, default=0.1, 
                        help="weight decay")
    parser.add_argument('--lradj', type = str, default = 'type1', 
                        help = 'adjust learning rate')
    parser.add_argument("--patience", type=int, required=True, default=7, 
                        help="early stopping patience")
    parser.add_argument("--checkpoints", type=str, required=False, 
                        default="./saved_results/pretrained_models/", 
                        help="checkpoints path")
    parser.add_argument("--test_results", type=str, required=False, default="./saved_results/test_results/",
                        help="test results path")
    parser.add_argument("--use_amp", type=int, required=True, default=1,
                        help="Use amp")
    # model pretrain device params
    parser.add_argument("--use_gpu", type=int, required=True, default=1, 
                        help="user gpu")
    parser.add_argument("--use_multi_gpu", type=int, required=True, default=0, 
                        help="use multi gpu")
    parser.add_argument("--gpu_type", type=str, required=True, default="cuda", 
                        help="gpu type")
    parser.add_argument("--devices", type=str, required=True, default="0,1,2,3",
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
    if args.task_name == 'tiny_gpt_pretrain':
        Exp = Model_Pretrain
    else:
        Exp = Model_Pretrain
    # ------------------------------
    # 模型训练
    # ------------------------------
    if args.is_train:
        for itr in range(args.iters):
            logger.info(f"{50 * '='}")
            logger.info(f"training iter: {itr}")
            logger.info(f"{50 * '='}")

            # setting record of experiments
            setting = f"{args.task_name}_{args.model_name}_{args.data_source.split('/')[-1][:-4]}_cl{args.context_length}_te{args.train_epochs}_bs{args.batch_size}"

            logger.info(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # set experiments
            exp = Exp(args)

            # model training
            exp.train(training_iter = itr, setting=setting, eval_freq=5, eval_iter=1, start_context="Every effort moves you")

            # model testing
            # logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # exp.test(setting)

            # empty cache
            torch.cuda.empty_cache()
    
    """
    # ------------------------------
    # 模型测试
    # ------------------------------
    if args.is_testing:
        ii = 0
        # setting record of experiments
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features} \
                    _sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model} \
                    _nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_expand{args.expand}_dc{args.d_conv} \
                    _fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}"
        # set experiments
        exp = Exp(args)
        logger.info(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test = 1)
        torch.cuda.empty_cache()
    
    # ------------------------------
    # 模型推理预测
    # ------------------------------
    if args.is_inference:
        logger.info(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        prediction = exp.predict(setting, True)
        torch.cuda.empty_cache()
        logger.info(prediction.shape)
    """




# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed()
    # 参数解析
    args = args_parse()
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
