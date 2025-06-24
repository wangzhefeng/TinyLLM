# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-24
# * Version     : 1.0.062422
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
import time
import urllib.request
import tarfile
from packaging import version
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.0**2 * duration)
    percent = count * block_size * 100.0 / total_size

    sys.stdout.write(
        f"\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB "
        f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
    )
    sys.stdout.flush()


def data_download(data_path: str="./dataset/finetune/adapter"):
    # create data path
    data_path = Path(data_path)
    os.makedirs(data_path, exist_ok=True)
    # data path
    source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target_file = data_path.joinpath("aclImdb_v1.tar.gz")
    target_path = data_path.joinpath("aclImdb")

    if target_file.exists():
        target_file.unlink()
    
    if not target_path.is_dir() and not target_file.is_file():
        urllib.request.urlretrieve(source, target_file, _reporthook)
    
    if not target_path.is_dir():
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall(data_path)


def load_dataset_into_to_dataframe(data_path: str="./dataset/finetune/adapter"):
    basepath = Path(data_path).joinpath("aclImdb")
    labels = {"pos": 1, "neg": 0}

    df = pd.DataFrame()
    with tqdm(total = 50000) as pbar:
        for s in ("test", "train"):
            for l in ("pos", "neg"):
                path = basepath.joinpath(s).joinpath(l)
                for file in sorted(path.iterdir()):
                    with open(file, "r", encoding="utf-8") as infile:
                        txt = infile.read()

                    if version.parse(pd.__version__) >= version.parse("1.3.2"):
                        x = pd.DataFrame(
                            [[txt, labels[l]]], 
                            columns=["review", "sentiment"]
                        )
                        df = pd.concat([df, x], ignore_index=False)
                    else:
                        df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()
    df.columns = ["text", "label"]
    
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    logger.info("Class distribution:")
    np.bincount(df["label"].values)

    return df


def partition_dataset(df, data_path: str="./dataset/finetune/adapter"):
    # data split
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()
    df_train = df_shuffled.iloc[:35_000]
    df_val = df_shuffled.iloc[35_000:40_000]
    df_test = df_shuffled.iloc[40_000:]
    # data save
    data_path = Path(data_path)
    df_train_path = data_path.joinpath("train.csv")
    df_val_path = data_path.joinpath("val.csv")
    df_test_path = data_path.joinpath("test.csv")
    if not df_train_path.exists():
        df_train.to_csv(data_path.joinpath("train.csv"), index=False, encoding="utf-8")
    if not df_val_path.exists():
        df_val.to_csv(data_path.joinpath("val.csv"), index=False, encoding="utf-8")
    if not df_test_path.exists():
        df_test.to_csv(data_path.joinpath("test.csv"), index=False, encoding="utf-8")
    
    return df_train, df_val, df_test


class IMDBDataset(Dataset):
    
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows




# 测试代码 main 函数
def main():
    data_path = "./dataset/finetune/adapter"

    # data download
    data_download(data_path)

    # data load
    # df = load_dataset_into_to_dataframe(data_path)

    # data partition
    # df_train, df_val, df_test = partition_dataset(df, data_path)
    
    # data read
    df_train = pd.read_csv(Path(data_path).joinpath("train.csv"))
    df_val = pd.read_csv(Path(data_path).joinpath("val.csv"))
    df_test = pd.read_csv(Path(data_path).joinpath("test.csv"))
    logger.info(f"df_train: \n{df_train.head()}")
    logger.info(f"df_val: \n{df_val.head()}")
    logger.info(f"df_test: \n{df_test.head()}")

if __name__ == "__main__":
    main()
