# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DistilBERT_sentiment_classification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-24
# * Version     : 1.0.062414
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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from datasets import load_dataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoModelForSequenceClassification

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


# ------------------------------
# dataset
# ------------------------------


# ------------------------------
# finetuning baseline: 
# finetuning the last layers of a DistilBERT model on a movie review dataset
# ------------------------------
# model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", 
    num_labels=2
)
logger.info(f"model architecture: \n{model}")




def experiments_compare_bar(accuracy, training_time):
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
