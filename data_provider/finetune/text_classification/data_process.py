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

from data_provider.finetune.text_classification.data_load import load_data
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def create_balanced_dataset(df):
    """
    create a balanced dataset
    """
    # count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    # randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    
    return balanced_df


def random_split(df, train_frac, valid_frac):
    """
    split dataframe into train, valid, test

    Args:
        df (_type_): _description_
        train_frac (_type_): _description_
        valid_frac (_type_): _description_

    Returns:
        _type_: _description_
    """
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


def data_to_csv(data_path, train_df, valid_df, test_df):
    """
    save data to csv

    Args:
        train_df (_type_): _description_
        valid_df (_type_): _description_
        test_df (_type_): _description_
    """
    # data dir
    os.makedirs(data_path, exist_ok=True)
    data_map = {
        "train.csv": train_df,
        "valid.csv": valid_df,
        "test.csv": test_df,
    }
    # data file path
    for data_name, data_obj in data_map.items():
        data_path = os.path.join(data_path, data_name)
        if not os.path.exists(data_path):
            data_obj.to_csv(data_path, index=None)
            logger.info(f"{data_name} saved to {data_path}")
        logger.info(f"{data_name} exists. Skipping save.")




# 测试代码 main 函数
def main():
    # data path
    data_dir = "dataset\finetuning\sms_spam_collection"

    # data load
    data_path = os.path.join(data_dir, "SMSSpamCollection.tsv")
    df = load_data(data_path)
    logger.info(f"df: \n{df.head()} \ndf.shape: {df.shape}")
    
    # create balanced dataset
    balanced_df = create_balanced_dataset(df)
    logger.info(f"balanced_df: \n{balanced_df.head()} \nbalanced_df.shape: {balanced_df.shape}")
    logger.info(f"balanced_df['Label'].value_counts(): \n{balanced_df['Label'].value_counts()}")
    
    # data split
    train_df, valid_df, test_df = random_split(balanced_df, train_frac=0.7, valid_frac=0.1)
    data_to_csv(data_dir, train_df, valid_df, test_df)

if __name__ == "__main__":
    main()
