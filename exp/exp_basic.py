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

from models import gpt
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Exp_Basic(object):
    
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "gpt": gpt,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def _build_data(self):
        pass

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) \
                if not self.args.use_multi_gpu \
                else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            logger.info('Use GPU: mps')
        else:
            device = torch.device('cpu')
            logger.info('Use CPU')
        
        return device 

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
