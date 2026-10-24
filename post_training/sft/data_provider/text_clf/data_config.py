# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-05
# * Version     : 0.1.030522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# zip file path
zip_data_path = Path("dataset/finetuning/sms_spam_collection.zip")
logger.info(f"zip_path: {zip_data_path}")

# data dir（惰性创建：仅在使用方真正取用时 mkdir，import 本模块无副作用）
_data_dir = Path("dataset/finetuning/sms_spam_collection")


def get_data_dir():
    _data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"data_dir: {_data_dir}")
    return _data_dir


# data file path（不触发目录创建）
tsv_file_path = Path(_data_dir).joinpath("SMSSpamCollection.tsv")
logger.info(f"tsv_file_path: {tsv_file_path}")


# 向后兼容：消费方以 data_config.data_dir 取用时才创建目录
def __getattr__(name):
    if name == "data_dir":
        return get_data_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
