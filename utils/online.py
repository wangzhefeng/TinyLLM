# -*- coding: utf-8 -*-

# ***************************************************
# * File        : online.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2026-10-25
# * Version     : 1.0.102510
# * Description : 训练脚本所需的在线服务登录（Hugging Face Hub / wandb）。
# *               token 优先级：函数参数 > 环境变量 > scripts/config.json；
# *               依赖懒加载：库缺失时仅告警并跳过，不中断训练主流程。
# * Link        : link
# * Requirement : 可选依赖 huggingface_hub / wandb
# ***************************************************

__all__ = [
    "hf_hub_login",
    "wandb_login",
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import json

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

# 私有配置文件（模板见 scripts/config.example.json，不进 git）
CONFIG_JSON_PATH = Path(ROOT).joinpath("scripts", "config.json")


def _load_config_token(key):
    """从 scripts/config.json 读取 token 字段；文件缺失或字段不存在返回 None。"""
    try:
        with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return config.get(key)


def hf_hub_login(token: str = None):
    """Hugging Face Hub 登录（可选依赖：huggingface_hub）。

    token 优先级：函数参数 > 环境变量 ``HF_TOKEN`` > scripts/config.json 的
    ``HF_ACCESS_TOKEN``；均未提供则跳过登录。
    """
    try:
        from huggingface_hub import login
    except ImportError:
        logger.warning("huggingface_hub 未安装，跳过 Hugging Face 登录（pip install huggingface_hub）")
        return
    token = token or os.environ.get("HF_TOKEN") or _load_config_token("HF_ACCESS_TOKEN")
    if not token:
        logger.warning("未找到 Hugging Face token（参数 / HF_TOKEN / HF_ACCESS_TOKEN 均为空），跳过登录")
        return
    login(token=token)
    logger.info("Hugging Face Hub 登录成功")


def wandb_login(api_key: str = None):
    """wandb 登录（可选依赖：wandb）。

    api key 优先级：函数参数 > 环境变量 ``WANDB_API_KEY`` > scripts/config.json 的
    ``WANDB_API_KEY``；均未提供则跳过登录。
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb 未安装，跳过 wandb 登录（pip install wandb）")
        return
    api_key = api_key or os.environ.get("WANDB_API_KEY") or _load_config_token("WANDB_API_KEY")
    if not api_key:
        logger.warning("未找到 wandb api key（参数 / WANDB_API_KEY 均为空），跳过登录")
        return
    wandb.login(key=api_key)
    logger.info("wandb 登录成功")


# 测试代码 main 函数
def main():
    hf_hub_login()
    wandb_login()

if __name__ == "__main__":
    main()
