# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_preprocessing.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-29
# * Version     : 1.0.012912
# * Description : description
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

import pandas as pd

from TinyLLM.data_provider.data_load_pretrain import load_spam_data
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def create_balanced_dataset(df):
    # count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    # randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    return balanced_df


def random_split(df, train_frac, valid_frac):
    # shuffle the entire dataframe
    df = df.sample(frac = 1, random_state = 123).reset_index(drop = True)
    # calculate split indices
    train_end = int(len(df) * train_frac)
    valid_end = train_end + int(len(df) * valid_frac)
    # split dataframe
    train_df = df[:train_end]
    valid_df = df[train_end:valid_end]
    test_df = df[valid_end:]
    
    return train_df, valid_df, test_df


def data_to_csv(train_df, valid_df, test_df):
    extracted_path = os.path.join(ROOT, "dataset/sms_spam_collection")
    train_df.to_csv(os.path.join(extracted_path, "train.csv"), index = None)
    valid_df.to_csv(os.path.join(extracted_path, "valid.csv"), index = None)
    test_df.to_csv(os.path.join(extracted_path, "test.csv"), index = None)




# 测试代码 main 函数
def main():
    df = load_spam_data()
    logger.info(f"df: \n{df.head()}")
    balanced_df = create_balanced_dataset(df)
    # logger.info(f"balanced_df: {balanced_df.head()}")
    logger.info(f"balanced_df.shape: {balanced_df.shape}")
    logger.info(f"balanced_df: \n{balanced_df['Label'].value_counts()}")
    # data split
    train_df, valid_df, test_df = random_split(balanced_df, 0.7, 0.1)
    data_to_csv(train_df, valid_df, test_df)

if __name__ == "__main__":
    main()
