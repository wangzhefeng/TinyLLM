# -*- coding: utf-8 -*-

# ***************************************************
# * File        : dapt_data.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101800
# * Description : 构建 DAPT（领域自适应预训练）数据：把下游任务（情感分类）训练集的
#                 无标注文本切句并构造 NSP 句子对，供 run_bert.py 继续 MLM 预训练。
#                 注意只用 train split，避免 valid/test 泄漏。
# * Link        : https://arxiv.org/abs/2004.10964
# * Requirement : pandas, tqdm
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse

from data_provider.bert.data_processor import DataProcessor
from post_training.sft.bert_sentiment.data import load_sentiment_file
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    parser = argparse.ArgumentParser(description="Build DAPT sentence-pair data from downstream unlabeled texts")
    parser.add_argument("--data_dir", type=str, default="./dataset/finetuning/sentiment", help="downstream data dir")
    parser.add_argument("--train_file", type=str, default="sentiment.train.data", help="train split file (texts only, labels ignored)")
    parser.add_argument("--output_task_file", type=str, default="task2_sentiment.csv", help="output NSP pair csv")
    return parser.parse_args()



# 测试代码 main 函数
def main():
    args = args_parse()
    # 只取 train split 的文本（标签丢弃，DAPT 是无标注预训练）
    texts, _ = load_sentiment_file(str(Path(args.data_dir) / args.train_file))
    logger.info(f"DAPT source texts: {len(texts)}")
    # 复用 THUCNews 的切句 + 正负样本构造逻辑
    processor = DataProcessor(data_dir=args.data_dir, output_task_file=args.output_task_file)
    processor.build_task2_dataset(text_list=texts)
    logger.info(f"DAPT sentence-pair data written to {Path(args.data_dir) / args.output_task_file}")

if __name__ == "__main__":
    main()
