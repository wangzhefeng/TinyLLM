# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_basic.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-13
# * Version     : 0.1.021322
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

import torch

from models import (
    gpt2_124M, 
    llama2, 
    llama3_8B,
)
from utils.ddp_utils import get_env_rank, get_env_world_size
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Exp_Basic:
    
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "gpt2_124M": gpt2_124M,
            "llama2": llama2,
            "llama3_8B": llama3_8B,
        }
        # device
        self.device = self._acquire_device()
        # tokenizer
        self.tokenizer = self._get_tokenizer()
        # model
        self.model = self._build_model()
        self.model = torch.compile(self.model)
        self.model.to(self.device).to(args.dtype)

    def _acquire_device(self):
        # Use GPU or not
        self.args.use_gpu = True if self.args.use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()) else False
        # GPU type: "cuda", "mps"
        self.args.gpu_type = self.args.gpu_type.lower().strip()
        # GPU device ids list
        self.args.devices = self.args.devices.replace(" ", "")
        self.args.device_ids = [int(id_) for id_ in self.args.devices.split(",")] 
        # device 
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            # GPU device ids
            device_id = get_env_rank()
            world_size = get_env_world_size()
            if not self.args.use_multi_gpu:
                if world_size == 1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = 0
                    device = torch.device(f"cuda:{0}")
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
                    device = torch.device(f"cuda:{device_id}")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device(f"cuda:{self.args.device_ids[0]}")
            logger.info(f"\t\tUse device GPU: {device}")
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
            logger.info(f"\t\tUse device GPU: mps")
        else:
            device = torch.device("cpu")
            logger.info("\t\tUse device CPU")

        return device
    
    def _get_data(self):
        pass
    
    def _get_tokenizer(self):
        pass 
    
    def _build_model(self):
        raise NotImplementedError
        return None 
    
    def train(self):
        pass 

    def vali(self):
        pass

    def inference(self):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
