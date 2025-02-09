# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_process.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-09
# * Version     : 0.1.020917
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
import csv
import json
import ujson
import jsonlines
import itertools
import psutil

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


bos_token = "<s>"
eos_token = "</s>"


def pretrain_process(data_path = "./dataset/pretrain/mobvoi_seq_monkey_general_open_corpus.jsonl", 
                     processed_data_path = "./dataset/pretrain/pretrain_data.csv",
                     chunk_size = 50000):
    chunk_idx = 0
    with jsonlines.open(data_path) as reader:
        with open(processed_data_path, "w", newline = "", encoding = "utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["text"])
            # 循环读取 JSONL 文件
            while True:
                # 读取 chunk_size 条数据
                chunk = list(itertools.islice(reader, chunk_size))
                if not chunk:
                    break
                # 处理每条数据
                for idx, obj in enumerate(chunk):
                    try:
                        content = obj.get("text", "")
                        if len(content) > 512:
                            continue
                        writer.writerow([content])
                    except UnicodeDecodeError as e:
                        logger.info(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                        continue
                chunk_idx += 1
                logger.info(f"chunk: {((chunk_idx - 1) * chunk_size, chunk_idx * chunk_size)}, process end")


def _chinese_ratio(text):
    """
    计算中文占比
    """
    # 匹配所有中文字符
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    # 中文字数占比
    chinese_ratio = len(chinese_chars) / len(text) if text else 0

    return chinese_ratio
    

def _process_and_write_data(data, contain_history, file_name):
    """
    process and write data

    Args:
        data (_type_): _description_
        contain_history (_type_): _description_
        file_name (_type_): _description_
    """
    q_list, a_list, history_list = [], [], []
    for per in data:
        history, q, a = per["history"], per["q"], per["a"]
        # TODO
        if (contain_history and not history) or not q or not a:
            continue
        # TODO
        if len(q) < 10 or len(a) < 5:
            continue
        # TODO
        if len(q) > 512 or len(a) > 512:
            continue
        # 判断 q 和 a 中文字符占比是否超过 70%
        if not (_chinese_ratio(q) > 0.5 and _chinese_ratio(a) > 0.5):
            continue
        # 数据收集
        q_list.append(q)
        a_list.append(a)
        if contain_history:
            history_list.append(history)
        else:
            history_list.append([])
    # 创建 DataFrame 并追加到 CSV 文件
    df = pd.DataFrame({
        "history": history_list,
        "q": q_list,
        "a": a_list,
    })
    try:
        df.to_csv(
            f"./dataset/{file_name}", mode = "a", header = False, index = False, 
            lineterminator = "\r\n", encoding = "utf-8"
        )
    except:
        # 若遇到数据 `_csv.Error: need to escape, but no escapechar set` 问题，可加 escapechar='\\' 参数
        df.to_csv(
            f"./dataset/{file_name}", mode = "a", header = False, index = False,
            lineterminator = "\r\n", escapechar = "\\", encoding = "utf-8",
        )


def sft_process(contain_history = False):
    # file name
    file_name = "sft_data_single.csv" if not contain_history else "sft_data.csv"
    # 每次处理的记录数
    chunk_size = 1000
    # 数据收集
    data = []
    with open(f"./dataset/{file_name}", "w", encoding = "utf-8") as f:
        f.write("history,q,a\n")
    
    sft_datasets = ["./dataset/sft_data_zh.jsonl"]
    if not contain_history:
        sft_datasets = ["./dataset/sft_data_zh.jsonl"]
    chunk_num = 0
    for path in sft_datasets:
        with jsonlines.open(path) as reader:
            for idx, obj in enumerate(reader):
                try:
                    data.append({
                        "history": obj.get("history", ""),
                        "q": obj.get("input", "") + obj.get("q", ""),
                        "a": obj.get("output", "") + obj.get("a", "")
                    })
                    if len(data) == chunk_size:
                        chunk_num += 1
                        _process_and_write_data(data)
                        data = []
                        if chunk_num % 100 == 0:
                            logger.info(f"chunk: {chunk_num} process end")
                except jsonlines.InvalidLineError as e:
                    logger.info(f"Skipping invalid JSON line {idx + 1}: {e}")
                    continue
            if data:
                _process_and_write_data(data)
                data = []


def rl_process():
    """
    DPO training
    """
    # training dataset
    dataset_paths = [
        "./dataset/dpo/dpo_zh_demo.json",
        "./dataset/dpo/dpo_train_data.json",
        "./dataset/dpo/huozi_rlhf_data.json",
    ]
    train_dataset = load_dataset("json", data_files = dataset_paths)

    merged_data = []
    for split in train_dataset.keys():
        merged_data.extend(train_dataset[split])
    
    with open("./dataset/dpo/train_data.json", "w", encoding = "utf-8") as f:
        json.dump(merged_data, f, ensure_ascii = False, indent = 4)




# 测试代码 main 函数
def main():
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./dataset/tokenizer", use_fast = False)
    logger.info(f"tokenizer vocab size: {len(tokenizer)}")

    # 1: pretrain
    # 2: sft
    # 3: RL
    process_type = 1

    if process_type == 1:
        pretrain_process()
    elif process_type == 2:
        sft_process()
    elif process_type == 3:
        rl_process()

if __name__ == "__main__":
    main()
