# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-05
# * Version     : 0.1.030522
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


from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# zip file path
zip_data_path = Path("dataset/finetune/sms_spam_collection.zip")
logger.info(f"zip_path: {zip_data_path}")

# data dir
data_dir = Path("dataset/finetune/sms_spam_collection")
os.makedirs(data_dir, exist_ok=True)
logger.info(f"data_dir: {data_dir}")

# data file path
tsv_file_path = Path(data_dir).joinpath("SMSSpamCollection.tsv")
logger.info(f"tsv_file_path: {tsv_file_path}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
