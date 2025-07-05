# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-05
# * Version     : 1.0.070522
# * Description : description
# * Link        : https://github.com/rasbt/LLM-finetuning-scripts/blob/main/conventional/distilbert-movie-review/1_feature-extractor.ipynb
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
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from data_provider.finetune.adapter.data_loader import IMDBDataset
from utils.device import device_setting
device = device_setting(True)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ------------------------------
# dataset
# ------------------------------
logger.info("Data Preparing...")
imdb_dataset = load_dataset(
    "./dataset/finetune/adapter",
    data_files = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }
)
logger.info(f"imdb_dataset: \n{imdb_dataset}")

# ------------------------------
# tokenize dataset
# ------------------------------
logger.info(f"Load tokenizer...")
# tokenzier
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
logger.info(f"Tokenizer input max length: {tokenizer.model_max_length}")
logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

# tokenize
def tokenize_text(batch):
    return tokenizer(
        batch["text"], 
        truncation=True, 
        padding=True
    )

imdb_tokenized = imdb_dataset.map(tokenize_text, batched = True, batch_size=None)
del imdb_dataset
imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------
# using DistilBERT as a Feature Extractor
# ------------------------------
logger.info(f"Load Model...")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.to(device)

# test --------------------------------
# test_batch = {
#     "attention_mask": imdb_tokenized["train"][:3]["attention_mask"].to(device),
#     "input_ids": imdb_tokenized["train"][:3]["input_ids"].to(device),
# }
# with torch.inference_mode():
#     test_output = model(**test_batch)
# logger.info(test_output.last_hidden_state.shape)

# cls_token_output = test_output.last_hidden_state[:, 0]
# cls_token_output.shape
# test --------------------------------

@torch.inference_mode()
def get_output_embeddings(batch):
    output = model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device)
    ).last_hidden_state[:, 0]

    return {"features": output.cpu().numpy()}


# extracted features
imdb_features = imdb_tokenized.map(get_output_embeddings, batched=True, batch_size=10)
# train
X_train = np.array(imdb_features["train"]["features"])
y_train = np.array(imdb_features["train"]["label"])
# valid
X_valid = np.array(imdb_features["validation"]["features"])
y_valid = np.array(imdb_features["validation"]["label"])
# test
X_test = np.array(imdb_features["test"]["features"])
y_test = np.array(imdb_features["test"]["label"])

# ------------------------------
# train model on embeddings(extracted features)
# ------------------------------
# model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# model evaluate
logger.info(f"train acc: {clf.score(X_train, y_train)}")
logger.info(f"valid acc: {clf.score(X_valid, y_valid)}")
logger.info(f"test acc: {clf.score(X_test, y_test)}")

# ------------------------------
# train random forestmodel on embeddings(extracted features)
# ------------------------------
# model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# model evaluate
print("Training accuracy", clf.score(X_train, y_train))
print("Validation accuracy", clf.score(X_valid, y_valid))
print("test accuracy", clf.score(X_test, y_test))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
