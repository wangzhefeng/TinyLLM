# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_gpt_instruction_sft.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-23
# * Version     : 0.1.022300
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

from exp.exp_finetune_gpt_instruction_flow import ModelFinetuningInstructionFlow
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
    parser.add_argument("--task_name", type=str, required=True, default="tiny_gpt_instruction_sft",
                        help="task name")
    parser.add_argument("--model_name", type=str, required=True, default="gpt_finetune_instruction",
                        help="model name")
    parser.add_argument("--is_training", type=int, required=True, default=1,
                        help="training flag")
    parser.add_argument("--is_inference", type=int, required=True, default=0,
                        help="inference flag")
    # data params
    parser.add_argument("--data_source", type=str, required=True, 
                        default="./dataset/finetune/instruction_data.json", 
                        help="data download url")
    parser.add_argument("--context_length", type=int, required=True, default=1024,
                        help="context length")
    # model params
    parser.add_argument("--train_ratio", type=float, required=True, default=0.85,
                        help="train data ratio")
    parser.add_argument("--test_ratio", type=float, required=True, default=0.10,
                        help="test data ratio") 
    parser.add_argument("--vocab_size", type=int, required=True, default=50257,
                        help="vocab size")
    parser.add_argument("--emb_dim", type=int, required=True, default=768,
                        help="embedding dimension")
    parser.add_argument("--n_heads", type=int, required=True, default=12,
                        help="number of heads")
    parser.add_argument("--n_layers", type=int, required=True, default=12,  
                        help="number of layers")
    parser.add_argument("--dropout", type=float, required=True, default=0.0, 
                        help="dropout")
    parser.add_argument("--qkv_bias", type=int, required=True, default=1, 
                        help="use bias in qkv")
    # model pretrain params
    parser.add_argument("--model_path", type=str, default="./saved_results/pretrained_models/tiny_gpt_pretrain_gpt_the-verdict_cl256_te10_bs2/checkpoint.pth",
                        help="model path")
    parser.add_argument("--pretrained_model", type=str, required=True, default="gpt2-small (124)",
                        help="pretrained model")
    parser.add_argument("--pretrained_model_path", type=str, required=True, default="./downloaded_models/gpt2_model",
                        help="pretrained model path")
    parser.add_argument("--pretrained_model_source", type=str, required=True, default="huggingface_gpt2",
                        help="pretrained model source")
    parser.add_argument("--finetuned_model_path", type=str, required=True, default="./saved_results/finetuning_pretrained_models",
                        help="finetuned model path")
    parser.add_argument("--tokenizer_model", type=str, required=True, default="gpt2",
                        help="tokenizer model")
    parser.add_argument("--seed", type=int, required=True, default=123,
                        help="random seed")
    parser.add_argument("--iters", type=int, required=True, default=10, 
                        help="number of iterations")
    parser.add_argument("--train_epochs", type=int, required=True, default=10, 
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True, default=8, 
                        help="batch size") 
    parser.add_argument("--learning_rate", type=float, required=True, default=5e-5, 
                        help="learning rate")
    parser.add_argument("--initial_lr", type=float, default=3e-5, 
                        help="initial learning rate") 
    parser.add_argument("--weight_decay", type=float, required=True, default=0.1, 
                        help="weight decay")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="minimum learning rate")
    parser.add_argument('--lradj', type = str, default = 'type1', 
                        help = 'adjust learning rate')
    parser.add_argument("--patience", type=int, default=7, 
                        help="early stopping patience")
    parser.add_argument("--checkpoints", type=str, 
                        default="./saved_results/pretrained_models/", 
                        help="checkpoints path")
    parser.add_argument("--test_results", type=str, default="./saved_results/test_results",
                        help="test results path")
    parser.add_argument("--use_amp", type=int, default=1,
                        help="Use amp")
    # model pretrain device params
    parser.add_argument("--num_workers", type=int, required=True, default=0,
                        help="num_workers")
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
    if args.task_name == 'tiny_gpt_instruction_sft':
        Exp = ModelFinetuningInstructionFlow
    else:
        Exp = ModelFinetuningInstructionFlow
    # ------------------------------
    # 模型训练
    # ------------------------------
    if args.is_training:
        for itr in range(args.iters):
            logger.info(f"{50 * '='}")
            logger.info(f"training iter: {itr}")
            logger.info(f"{50 * '='}")

            # setting record of experiments
            setting = f"{args.task_name}_{args.model_name}_{args.data_source.split('/')[-1][:-6]}_cl{args.context_length}_te{args.train_epochs}_bs{args.batch_size}"

            logger.info(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # set experiments
            exp = Exp(args)

            # model training
            exp.train(training_iter = itr, setting = setting, eval_freq=5, eval_iter=5)

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
    # 参数解析
    args = args_parse()
    # 设置随机数
    set_seed(args.seed)
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
