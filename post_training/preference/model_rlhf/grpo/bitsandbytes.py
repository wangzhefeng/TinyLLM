# -*- coding: utf-8 -*-

# ***************************************************
# * File        : bitsandbytes.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-20
# * Version     : 1.0.072012
# * Description : 定义 bitsandbytes 中的 mock 类，用于绕过安装依赖
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import types
import importlib.machinery
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# bnb
bnb_name = "bitsandbytes"
spec = importlib.machinery.ModuleSpec(name=bnb_name, loader=None, is_package=True)
bnb = types.ModuleType(bnb_name)
bnb.__spec__ = spec
bnb.__path__ = []
bnb.__version__ = "0.0.0"

# bnb_nn
nn_name = bnb_name + ".nn"
nn_spec = importlib.machinery.ModuleSpec(name=nn_name, loader=None, is_package=True)
bnb_nn = types.ModuleType(nn_name)
bnb_nn.__spec__ = nn_spec
bnb_nn.__path__ = []

class Linear4bit: pass
class Linear8bitLt: pass
bnb_nn.Linear4bit = Linear4bit
bnb_nn.Linear8bitLt = Linear8bitLt
bnb.nn = bnb_nn

sys.modules[bnb_name] = bnb
sys.modules[nn_name] = bnb_nn




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
