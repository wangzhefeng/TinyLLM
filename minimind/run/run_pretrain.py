# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_pretrain.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-05-03
# * Version     : 1.0.050302
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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def arg_parser():
    """
    user command setting
    """
    parser = argparse.ArgumentParser(description="Pretraining")
    # basic config
    parser.add_argument("--is_train", type=int, required=True, default=0, help='Whether to conduct train')
    parser.add_argument("--is_inference", type=int, required=True, default=0, help='Whether to conduct inference')
    parser.add_argument("--model_name", type=str, required=True, default='Transformer', help='model name')
    # data loader
    parser.add_argument("--root_path", type=str, required=True, default='./minimind/dataset/', help='root path of the data file')
    parser.add_argument("--data_path", type=str, required=True, default='pretrain_hp.jsonl', help='data file')
    parser.add_argument("--train_ratio", type=float, required=True, default=0.7, help='train dataset ratio')
    parser.add_argument("--test_ratio", type=float, required=True, default=0.2, help='test dataset ratio')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size of train input data')
    # model define
    # parser.add_argument("--dropout", type=float, required=True, default=0.0, help="dropout rate")
    # parser.add_argument("--bos_token_id", type=int, required=True, default=1, help="begin of sentence token id")
    # parser.add_argument("--eos_token_id", type=int, required=True, default=2, help="end of sentence token id")
    # parser.add_argument("--hidden_act", type=str, required=True, default="silu", help="hidden layer activation function")
    # parser.add_argument("--hidden_size", type=int, required=True, default=512, help="hidden layer size")
    # parser.add_argument("--intermediate_size", type=int, required=True, default=None, help="intermediate size")
    # parser.add_argument("--max_position_embeddings", type=int, required=True, default=32768, help="max position embeddings")
    # parser.add_argument("--num_attention_heads", type=int, required=True, default=8, help="number of attention heads")
    # parser.add_argument("--num_hidden_layers", type=int, required=True, default=8, help="number of hidden layers")
    # parser.add_argument("--num_key_value_heads", type=int, required=True, default=2, help="number of key and value heads")
    # parser.add_argument("--vocab_size", type=int, required=True, default=6400, help="vocab size")
    # parser.add_argument("--rms_norm_eps", type=float, required=True, default=1e-05, help="rms norm eps")
    # parser.add_argument("--rope_theta", type=int, required=True, default=1000000.0, help="rope theta")
    # parser.add_argument("--flash_attn", type=bool, required=True, default=True, help="flash attention")
    # parser.add_argument("--use_moe", type=bool, required=True, default=False, help="is use moe")
    # parser.add_argument("--num_experts_per_tok", type=int, required=True, default=2, help="number of experts per token")
    # parser.add_argument("--n_routed_experts", type=int, required=True, default=4, help="number of routed experts")
    # parser.add_argument("--n_shared_experts", type=int, required=True, default=1, help="number of shared experts")
    # parser.add_argument("--scoring_func", type=str, required=True, default="softmax", help="socring function")
    # parser.add_argument("--aux_loss_alpha", type=float, required=True, default=0.1, help="aux loss alpha")
    # parser.add_argument("--seq_aux", type=bool, required=True, default=True, help="seq aux")
    # parser.add_argument("--norm_topk_prob", type=bool, required=True, default=True, help="norm topk probablity")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    # wandb config
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="Pretrain", help="Weights & Biases project name")
    # data
    parser.add_argument("--data_path", type=str, default="", help="Path to training data")
    # training
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--checkpoints', type=str, default='./saved_results/pretrained_models/', help='location of model models')
    parser.add_argument('--test_results', type=str, default='./saved_results/test_results/', help='location of model models')
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--patience', type = int, default=3, help = 'early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', type=int, default=0, help='use automatic mixed precision training') 
    # GPU
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--device_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help = 'use multiple gpus')
    parser.add_argument('--devices', type=str, default="0,1,2,3,4,5,6,7,8", help='device ids of multile gpus')
    
    # 命令行参数解析
    args = parser.parse_args()

    return args




# 测试代码 main 函数
def main():
    args = arg_parser()
    
    torch.manual_seed(1337)
    wandb_run_name = f"Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

if __name__ == "__main__":
    main()
