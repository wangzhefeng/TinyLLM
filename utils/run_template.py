# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-08
# * Version     : 1.0.010821
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
from pathlib import Path

from exp.exp_forecasting import Exp_Forecast
from utils.args_tools import print_args
from utils.device import torch_gc
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    parser = argparse.ArgumentParser(description='LLM')
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='pretrain', help='task name')
    parser.add_argument('--is_training', type=int, required=True, default=0, help='Whether to conduct training')
    parser.add_argument('--is_testing', type=int, required=True, default=0, help='Whether to conduct testing')
    parser.add_argument('--is_inference', type=int, required=True, default=0, help='Whether to conduct inference')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer', help='model name')
    # data loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, default='ETTh1.csv', help='data file')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--checkpoints', type=str, default='./saved_results/pretrained_models/', help='location of model models')
    parser.add_argument('--test_results', type=str, default='./saved_results/test_results/', help='location of model models')
    parser.add_argument('--predict_results', type=str, default='./saved_results/predict_results/', help='location of model models') 
    # forecasting task
    parser.add_argument('--train_ratio', type=float, required=True, default=0.7, help='train dataset ratio')
    parser.add_argument('--test_ratio', type=float, required=True, default=0.2, help='test dataset ratio')
    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--iters', type=int, default=1, help='train iters')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type = int, default=3, help = 'early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', type=int, default=0, help='use automatic mixed precision training') 
    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help = 'use multiple gpus')
    parser.add_argument('--devices', type=str, default="0,1,2,3,4,5,6,7,8", help='device ids of multile gpus')
    
    # 命令行参数解析
    args = parser.parse_args()

    return args


def run(args):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_te{}_{}_{}_'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        args.train_epochs,
        args.root_path.split("/")[-2],
        args.target,
        # TODO args.add_fredf,
    )
    
    # 模型训练
    if args.is_training: 
        for ii in range(args.iters):
            # setting record of experiments
            training_setting = setting + str(ii)
            logger.info(f">>>>>>>>> start training: iter-{ii}: {training_setting}>>>>>>>>>>")
            logger.info(f"{180 * '='}")
            # 实例化
            exp = Exp_Forecast(args)
            # 模型训练
            model, train_results = exp.train(training_setting)
            # 模型测试
            if args.is_testing:
                logger.info(f">>>>>>>>> start testing: iter-{ii}: {training_setting}>>>>>>>>>>")
                logger.info(f"{180 * '='}")
                exp.test(flag="test", setting=training_setting, load=False)





# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed(seed = 2023)
    # 参数解析
    args = args_parse()
    print_args(args)
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
