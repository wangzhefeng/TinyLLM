# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_gpt_classification_finetuning.py
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
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse

import torch

from exp.exp_finetune_gpt_clf import ModelFinetuningClassifier
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def args_parse():
    # ------------------------------
    # parser
    # ------------------------------
    parser = argparse.ArgumentParser(description="Tiny GPT Finetuning Classification")
    # ------------------------------
    # add arguments
    # ------------------------------
    # task params
    parser.add_argument("--task_name", type=str, required=True, default="tiny_gpt_classification_sft",
                        help="task name")
    parser.add_argument("--model_name", type=str, required=True, default="gpt_finetune_clf",
                        help="model name")
    parser.add_argument("--is_train", type=int, required=True, default=1,
                        help="training flag")
    parser.add_argument("--is_inference", type=int, required=True, default=0,
                        help="inference flag")
    # data params
    parser.add_argument("--data_source", type=str, required=True, 
                        default="./dataset/finetune/sms_spam_collection", 
                        help="data download url")
    parser.add_argument("--context_length", type=int, required=True, default=1024,
                        help="context length")
    # model params
    parser.add_argument("--num_classes", type=int, required=True, default=2,
                        help="number of classes")
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
    parser.add_argument("--pretrained_model", type=str, required=True, default="gpt2-small (124)",
                        help="pretrained model")
    parser.add_argument("--pretrained_model_path", type=str, required=True, default="./downloaded_models/gpt2_model",
                        help="pretrained model path")
    parser.add_argument("--pretrained_model_source", type=str, required=True, default="huggingface_gpt2",
                        help="pretrained model source")
    parser.add_argument("--finetuned_model_path", type=str, required=True, default="./saved_results/finetuned_models",
                        help="finetuned model path")
    parser.add_argument("--finetune_method", type=str, required=True, default="simple",
                        help="finetune method")
    parser.add_argument("--tokenizer_model", type=str, required=True, default="gpt2",
                        help="tokenizer model")
    parser.add_argument("--seed", type=int, required=True, default=123,
                        help="random seed")
    parser.add_argument("--iters", type=int, required=True, default=10, 
                        help="number of iterations")
    parser.add_argument("--train_epochs", type=int, required=True, default=10, 
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, required=True, default=2, 
                        help="batch size") 
    parser.add_argument("--learning_rate", type=float, required=True, default=5e-5, 
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, required=True, default=0.1, 
                        help="weight decay")
    parser.add_argument("--initial_lr", type=float, default=3e-5, 
                        help="initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="minimum learning rate") 
    parser.add_argument('--lradj', type = str, default = 'type1', 
                        help = 'adjust learning rate')
    parser.add_argument("--patience", type=int, default=7, 
                        help="early stopping patience")
    parser.add_argument("--checkpoints", type=str, 
                        default="./saved_results/finetuned_models", 
                        help="checkpoints path")
    parser.add_argument("--test_results", type=str, default="./saved_results/test_results",
                        help="test results path")
    parser.add_argument("--use_amp", type=int, default=1,
                        help="Use amp")
    parser.add_argument("--num_workers", type=int, required=True, default=0,
                        help="num_workers")
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
    if args.task_name == 'tiny_gpt_classification_sft':
        Exp = ModelFinetuningClassifier
    else:
        Exp = ModelFinetuningClassifier
    # ------------------------------
    # 模型训练
    # ------------------------------
    if args.is_train:
        for itr in range(args.iters):
            logger.info(f"{50 * '='}")
            logger.info(f"training iter: {itr}")
            logger.info(f"{50 * '='}")
            # setting record of experiments
            setting = f"{args.task_name}_{args.model_name}_{args.data_source.split('/')[-1]}_cl{args.context_length}_te{args.train_epochs}_bs{args.batch_size}"
            logger.info(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # set experiments
            exp = Exp(args)
            # model training
            exp.train(training_iter=itr, setting=setting, eval_freq=50, eval_iter=5)
            # empty cache
            torch.cuda.empty_cache()
    # ------------------------------
    # 模型推理预测
    # ------------------------------
    if args.is_inference:
        # before finetune
        input_prompt = "Every effort moves you"
        # test input text 1
        text_1 = (
            "You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award."
        )
        # test input text2
        text_2 = (
            "Hey, just wanted to check if we're still on"
            " for dinner tonight? Let me know!"
        )
        logger.info(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # test inference on text 1
        logger.info(f"input_text: {text_1}")
        prediction = exp.inference(text = text_1)
        logger.info(f"prediction label: {prediction}")
        # test inference on text 2
        logger.info(f"input_text: {text_2}")
        prediction = exp.inference(text = text_2)
        logger.info(f"prediction: {prediction}")
        # empty cache
        torch.cuda.empty_cache() 




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
