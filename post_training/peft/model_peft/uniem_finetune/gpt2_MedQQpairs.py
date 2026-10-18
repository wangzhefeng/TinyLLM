# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt2_MedQQpairs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-21
# * Version     : 0.1.102122
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


# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


from datasets import load_dataset
from transformers import AutoTokenizer

from uniem.finetuner import FineTuner
from uniem.training_strategy import BitFitTrainging
from uniem.model import PoolingStrategy, create_uniem_embedder

dataset = load_dataset('vegaviazhang/Med_QQpairs')
dataset = dataset.rename_columns({'question1': 'sentence1', 'question2': 'sentence2'})

embedder = create_uniem_embedder('gpt2', pooling_strategy = PoolingStrategy.last_weighted)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

finetuner = FineTuner(embedder, tokenizer, dataset = dataset)
finetuner.tokenizer.pad_token = finetuner.tokenizer.eos_token
finetuner.run(
    epochs = 3, 
    lr = 1e-3, 
    batch_size = 32, 
    training_strategy = BitFitTrainging()
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
