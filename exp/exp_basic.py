# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_basic.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-13
# * Version     : 0.1.021322
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

import torch

from models import gpt, llama2
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Basic:
    
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "gpt": gpt,
            "llama2": llama2,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device) 
 
    def _acquire_device(self):
        # use gpu or not
        self.args.use_gpu = True \
            if self.args.use_gpu and (torch.cuda.is_available() or torch.backends.mps.is_available()) \
            else False
        # gpu type: "cuda", "mps"
        self.args.device_type = self.args.device_type.lower().strip()
        # gpu device ids strings
        self.args.devices = self.args.devices.replace(" ", "")
        # gpu device ids list
        self.args.device_ids = [int(id_) for id_ in self.args.devices.split(",")]
        # gpu device ids string
        if self.args.use_gpu and not self.args.use_multi_gpu:
            self.gpu = self.args.device_ids[0]  # '0'
        elif self.args.use_gpu and self.args.use_multi_gpu:
            self.gpu = self.args.devices  # '0,1,2,3,4,5,6,7'
        else:
            self.gpu = "0"
        
        # device
        if self.args.use_gpu and self.args.device_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.gpu}")
            logger.info(f"Use device GPU: cuda:{self.gpu}")
        elif self.args.use_gpu and self.args.device_type == "mps":
            device = torch.device("mps") \
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
                else torch.device("cpu")
            logger.info(f"Use device GPU: mps")
        else:
            device = torch.device("cpu")
            logger.info("Use device CPU")

        return device
    
    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _get_data(self):
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
