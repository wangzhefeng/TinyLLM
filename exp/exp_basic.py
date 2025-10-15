# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_basic.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-13
# * Version     : 0.1.021322
# * Description : description
# * Link        : Thunder: https://github.com/Lightning-AI/lightning-thunder
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path

from models.gpt2 import gpt2_124M
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import thunder

from models import (
    llama2, 
    llama3_8B,
)
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Exp_Basic:
    
    def __init__(self, args, local_rank):
        self.args = args
        self.model_dict = {
            "gpt2_124M": gpt2_124M,
            "llama2": llama2,
            "llama3_8B": llama3_8B,
        }
        # device
        self.device = self._acquire_device(local_rank)
        # tokenizer
        self.tokenizer = self._get_tokenizer()
        # model
        self.model = self._build_model()
        # model compile
        major, minor = map(int, torch.__version__.split(".")[:2])
        if self.args.compile and self.args.gpu_type == "cuda":
            if (major, minor) > (2, 8):
                # This avoids retriggering model recompilations in PyTorch 2.8 and newer
                # if the model contains code like self.pos = self.pos + 1
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
            if args.compile_type == "torch":
                self.model = torch.compile(self.model)
            elif args.compile_type == "thunder":
                self.model = thunder.compile(self.model) 
        # model device and dtype
        self.model.to(self.device).to(args.dtype)

    def _acquire_device(self, local_rank):
        # Use GPU or not
        self.args.use_gpu = True \
            if self.args.use_gpu and (
                torch.cuda.is_available() or 
                torch.backends.mps.is_available() or 
                torch.xpu.is_available()
            ) else False
        # GPU type: "cuda", "mps", "xpu"
        self.args.gpu_type = self.args.gpu_type.lower().strip()
        # GPU device ids list(CUDA_VISIBLE_DEVICES)
        self.args.device_ids = [int(id_) for id_ in os.environ["CUDA_VISIBLE_DEVICES"].replace(" ", "").split(",")]
        # device
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            if self.args.use_dp:
                device = torch.device(f"cuda:{self.args.device_ids[0]}")
            elif self.args.use_ddp:
                # 显式设置当前进程使用的 GPU 设备
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device("cuda", 0)
            logger.info(f"\t\tUse NVIDIA CUDA GPU: {device}")
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps") \
                if hasattr(torch.backends, "mps") and \
                torch.backends.mps.is_available() \
                else torch.device("cpu")
            logger.info(f"\t\tUse Apple Silicon GPU (MPS)")
        elif self.args.use_gpu and self.args.gpu_type == "xpu":
            device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
            logger.info(f"\t\tUse Intel GPU")
        else:
            device = torch.device("cpu")
            logger.info("\t\tUse CPU")

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
