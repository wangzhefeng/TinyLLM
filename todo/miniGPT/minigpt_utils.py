# -*- coding: utf-8 -*-

# ***************************************************
# * File        : utils.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-04
# * Version     : 1.0.070413
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json
import random
from typing import Dict
from ast import literal_eval

import numpy as np
import torch

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """
    monotonous bookkeeping
    """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """
    a lightweight configuration class inspired by yacs
    """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent: int=0):
        """
        need to have a helper to support nested indentation for pretty printing
        """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f"{k}: {v}\n")
        parts = [' ' * (indent * 4) + p for p in parts]
        
        return "".join(parts)

    def to_dict(self):
        """
        return a dict representation of the config
        """
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v 
            for k, v in self.__dict__.items() 
        }

    def merge_from_dict(self, d: Dict):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            logger.info("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)




# 测试代码 main 函数
def main():
    C = CfgNode()
    C.model_type = "gpt"
    C.n_layer = None
    C.n_head = None
    C.n_embd = None
    C.vocab_size = None
    C.block_size = None
    C.embd_pdrop = 0.1
    C.resid_pdrop = 0.1
    C.attn_pdrop = 0.1
    logger.info(f"C: \n{C} \nType of C: {type(C)}")
    
    config_print = C._str_helper(indent=0)
    logger.info(config_print)

    config_dict = C.to_dict()
    logger.info(config_dict)
    
    C.merge_from_dict({
        "test1": 1,
        "test2": 2,
        "test3": 3,
    })
    logger.info(f"C: \n{C} \nType of C: {type(C)}")

    # TODO test
    C.merge_from_args(['--model.n_layer=10', '--trainer.batch_size=32'])
    logger.info(f"C: \n{C} \nType of C: {type(C)}")

if __name__ == "__main__":
    main()
