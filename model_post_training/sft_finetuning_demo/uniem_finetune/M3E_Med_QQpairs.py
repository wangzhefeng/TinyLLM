# -*- coding: utf-8 -*-

# ***************************************************
# * File        : M3E_Med_QQpairs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-21
# * Version     : 0.1.102121
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


from datasets import load_dataset
from uniem.finetuner import FineTuner

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data load and preprocess
# ------------------------------
# 下载数据
dataset = load_dataset('vegaviazhang/Med_QQpairs')["train"]
# 查看一下 Med_QQpairs 的数据格式是不是在 FineTuner 支持的范围内
print(dataset[0])
print(dataset[1])
dataset = dataset.rename_columns({
    "question1": "sentence1",
    "question2": "sentence2",
})
# Med_QQpairs 只有训练集，需要手动划分训练集和验证集
dataset = dataset.train_test_split(test_size = 0.1, seed = 42)
dataset['validation'] = dataset.pop('test')
print(dataset)

# ------------------------------
# model finetune 
# ------------------------------
finetuner = FineTuner.from_pretrained(
    'moka-ai/m3e-small', 
    dataset = dataset
)
fintuned_model = finetuner.run(
    epochs = 3,
    output_dir = "E:/projects/llms_proj/llm_proj/embedding_api/finetuned-model/m3e-medqqpairs/"
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
