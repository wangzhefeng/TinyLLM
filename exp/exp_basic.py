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

from models import gpt, llama2, llama3_8B
from model_post_training.lora_qlora.scratch_exps.lora_dora_from_scratch import MLP
# from todo.minimind.model import model as minimind
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Exp_Basic:
    
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "gpt": gpt,
            "llama2": llama2,
            "llama3_8B": llama3_8B,
            # "minimind": minimind,
            "mlp": MLP,
        }
        self.device = self._acquire_device()
        self.tokenizer = self._get_tokenizer()
        self.model = self._build_model().to(self.device)
 
    def _acquire_device(self):
        # use gpu or not
        self.args.use_gpu = True \
            if self.args.use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()) \
            else False
        # gpu type: "cuda", "mps"
        self.args.gpu_type = self.args.gpu_type.lower().strip()
        # gpu device ids list
        self.args.devices = self.args.devices.replace(" ", "")
        self.args.device_ids = [int(id_) for id_ in self.args.devices.split(",")]
        # gpu device ids string
        self.gpu = self.args.device_ids[0]  # ro self.gpu = "0"
        # device
        if self.args.use_gpu and self.args.gpu_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.gpu}")
            logger.info(f"\t\tUse device GPU: cuda:{self.gpu}")
        elif self.args.use_gpu and self.args.gpu_type == "mps":
            device = torch.device("mps") \
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
                else torch.device("cpu")
            logger.info(f"\t\tUse device GPU: mps")
        else:
            device = torch.device("cpu")
            logger.info("\t\tUse device CPU")

        return device
    
    def _get_data(self):
        pass
    
    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _get_tokenizer(self):
        pass
    
    def train(self):
        pass 

    def vali(self):
        pass

    def test(self):
        pass

    def inference(self):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
