# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_load_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012912
# * Description : finetuning for text classification
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def download_and_unzip_spam_data(data_file_path, zip_data_path, extracted_path):
    """
    download spam data for finetuning text classification
    """
    # data file path check
    if data_file_path.exists():
        logger.info(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    # data file download
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    with urllib.request.urlopen(url) as response:
        with open(zip_data_path, "wb") as out_file:
            out_file.write(response.read())
    # unzipping the file
    with zipfile.ZipFile(zip_data_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    # add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    logger.info(f"File downloaded and saved as {data_file_path}")


def load_data(data_file_path):
    """
    load spam tsv data for finetuning text classification
    """
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    return df




# 测试代码 main 函数
def main():
    from data_provider.finetune.text_clf.data_config import (
        zip_data_path, 
        data_dir, 
        tsv_file_path
    )
    # data download
    download_and_unzip_spam_data(tsv_file_path, zip_data_path, data_dir)

    # data load
    df = load_data(tsv_file_path)
    logger.info(f"df: \n{df.head()} \ndf.shape: {df.shape}")
    logger.info(f"df['Label'].value_counts(): \n{df['Label'].value_counts()}")

if __name__ == "__main__":
    main()

