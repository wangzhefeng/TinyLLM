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
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


import pandas as pd

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


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


def data_to_csv(data_dir, train_df, valid_df, test_df):
    """
    save data to csv
    """
    # data name: file map
    data_map = {
        "train.csv": train_df,
        "valid.csv": valid_df,
        "test.csv": test_df,
    }
    # data file path
    for data_name, data_obj in data_map.items():
        data_path = Path(data_dir).joinpath(data_name)
        if not Path(data_path).exists():
            data_obj.to_csv(data_path, index = None)
            logger.info(f"{data_name} saved to {data_path}")
        logger.info(f"{data_name} exists. Skipping save")




# 测试代码 main 函数
def main():
    from data_provider.finetune.text_clf.data_config import data_dir, tsv_file_path
    from data_provider.load_save_data import load_csv_data

    # data load
    df = load_csv_data(data_file_path = tsv_file_path)
    logger.info(f"df: \n{df.head()} \ndf.shape: {df.shape}")
    logger.info(f"df['Label'].value_counts(): \n{df['Label'].value_counts()}")
    
    # create balanced dataset
    balanced_df = create_balanced_dataset(df = df)
    logger.info(f"balanced_df: \n{balanced_df.head()} \nbalanced_df.shape: {balanced_df.shape}")
    logger.info(f"balanced_df['Label'].value_counts(): \n{balanced_df['Label'].value_counts()}")
    
    # data split
    train_df, valid_df, test_df = random_split(
        df = balanced_df, 
        train_frac = 0.7, 
        valid_frac = 0.1
    )
    logger.info(f"train_df length: {len(train_df)}")
    logger.info(f"valid_df length: {len(valid_df)}")
    logger.info(f"test_df length: {len(test_df)}")
    data_to_csv(data_dir, train_df, valid_df, test_df)

if __name__ == "__main__":
    main()
