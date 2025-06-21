# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tools.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-13
# * Version     : 1.0.011322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DotDict(dict):

    """
    dot.notation access to dictionary attributes
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def print_args(args):
    # ------------------------------
    # Basic Config
    # ------------------------------
    logger.info(f'{100 * "-"}')
    logger.info(f'Args in experiment:')
    logger.info(f'{100 * "-"}')
    logger.info("\033[1m" + "Basic Config" + "\033[0m")
    logger.info(f'  {"Task Name:":<20}{args.task_name:<20}{"Description:":<20}{args.des:<20}')
    logger.info(f'  {"Is Training:":<20}{args.is_train:<20}{"Is Testing:":<20}{args.is_test:<20}')
    logger.info(f'  {"Is Inference:":<20}{args.is_inference:<20}{"Random Seed:":<20}{args.seed:<20}')
    logger.info(f'  {"Model:":<20}{args.model_name:<20}')
    logger.info("")
    # ------------------------------
    # Data Loader
    # ------------------------------
    logger.info("\033[1m" + "Data Loader" + "\033[0m")
    logger.info(f'  {"Data Name:":<20}{args.data_name:<20}{"Data Path:":<20}{args.data_path:<20}')
    logger.info(f'  {"Data File:":<20}{args.data_file:<20}{"Train Ratio:":<20}{args.train_ratio:<20}')
    logger.info(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    logger.info(f'  {"Test results:":<20}{args.test_results:<20}')
    logger.info("")
    # ------------------------------
    # Model Parameters
    # ------------------------------
    if args.task_name in ['tiny_gpt_pretrain']:
        logger.info("\033[1m" + "Forecasting Task" + "\033[0m")
        logger.info(f'  {"Context Lenght:":<20}{args.context_length:<20}{"Vocab Size:":<20}{args.vocab_size:<20}')
        logger.info(f'  {"Embedding Dim:":<20}{args.emb_dim:<20}{"Number Heads:":<20}{args.n_heads:<20}')
        logger.info(f'  {"Number Layers:":<20}{args.n_layers:<20}{"Dropout rate:":<20}{args.dropout:<20}')
        logger.info(f'  {"QKV Bias:":<20}{args.qkv_bias:<20}{"Dtype:":<20}{str(args.dtype):<20}')
        logger.info(f'  {"Max New Tokens:":<20}{args.max_new_tokens:<20}{"Tokenizer Model:":<20}{args.tokenizer_model:<20}')
        logger.info("") 
    # ------------------------------
    # Run Parameters
    # ------------------------------
    logger.info("\033[1m" + "Run Parameters" + "\033[0m")
    logger.info(f'  {"Train Iter:":<20}{args.iters:<20}{"Train Epochs:":<20}{args.train_epochs:<20}')
    logger.info(f'  {"Batch Size:":<20}{args.batch_size:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    logger.info(f'  {"Learning Rate:":<20}{args.learning_rate:<20}{"Initial Learning Rate:":<20}{args.initial_lr:<20}')
    logger.info(f'  {"Minimum Learning Rate:":<20}{args.min_lr:<20}{"Weight Decay:":<20}{args.weight_decay:<20}')
    logger.info(f'  {"Learning Rate Adjust:":<20}{args.lradj:<20}{"Patience:":<20}{args.patience:<20}')
    logger.info("")
    # ------------------------------
    # GPU
    # ------------------------------
    logger.info("\033[1m" + "GPU" + "\033[0m")
    logger.info(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU Type:":<20}{args.gpu_type:<20}')
    logger.info(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    logger.info(f'  {"Number Workders:":<20}{args.num_workers:<20}')
    logger.info("")
    # logger.info("")
    logger.info(f'{100 * "-"}')




# 测试代码 main 函数
def main():
    dct = {
        'scalar_value': 1, 
        'nested_dict': {
            'value': 2, 
            'nested_nested': {
                'x': 21
            }
        }
    }
    dct = DotDict(dct)

    print(dct.nested_dict.nested_nested.x)

if __name__ == "__main__":
    main()
