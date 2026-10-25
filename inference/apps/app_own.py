# -*- coding: utf-8 -*-

# ***************************************************
# * File        : app_own.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-09-07
# * Version     : 1.1.102000
# * Description : 【已合并到 app.py，本文件仅为兼容保留】
#               自训 checkpoint 的 Chainlit UI 入口。
#               等价于：GPT2_WEIGHTS=own chainlit run ./apps/app.py
#               （checkpoint 路径用环境变量 GPT2_CHECKPOINT 覆盖，默认 results/checkpoints/gpt2/model.pth）
#               兼容机制说明：chainlit 的 @chainlit.on_message 装饰器把 handler
#               注册到 chainlit 的全局回调（config.code.on_message），不绑定
#               入口文件本身，因此本 shim 只需设置 GPT2_WEIGHTS=own 后 import
#               app 模块，handler 即注册生效。
# * Link        : link
# * Requirement : pip install chainlit（可选依赖，不在主 requirements 中）
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path

# 选择自训 checkpoint（必须在 import app 之前设置）
os.environ["GPT2_WEIGHTS"] = "own"

# 让同目录的 app.py 可被 import（chainlit 不保证把入口文件目录加入 sys.path）
APP_DIR = str(Path(__file__).resolve().parent)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# import 即完成模型加载与 @chainlit.on_message handler 注册
from app import main  # noqa: F401

if __name__ == "__main__":
    main()
