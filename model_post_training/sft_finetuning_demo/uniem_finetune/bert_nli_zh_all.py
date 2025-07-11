# -*- coding: utf-8 -*-

# ***************************************************
# * File        : bert_nli_zh_all.py
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


from datasets import load_dataset
from transformers import AutoTokenizer

from uniem.finetuner import FineTuner
from uniem.model import create_uniem_embedder

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data load and preprocess
# ------------------------------
dataset = load_dataset('shibing624/nli-zh-all', streaming = True)
dataset = dataset.rename_columns({
    'text1': 'sentence1', 
    'text2': 'sentence2'
})

# ------------------------------
# model finetune
# ------------------------------
# 由于是从头训练，需要初始化 embedder 和 tokenizer。当然，也可以选择新的 pooling 策略。
embedder = create_uniem_embedder('uer/chinese_roberta_L-2_H-128', pooling_strategy = 'cls')
tokenizer = AutoTokenizer.from_pretrained('uer/chinese_roberta_L-2_H-128')
finetuner = FineTuner(embedder, tokenizer = tokenizer, dataset=dataset)
fintuned_model = finetuner.run(
    epochs = 3, 
    batch_size = 32, 
    lr = 1e-3,
    output_dir = "E:/projects/llms_proj/llm_proj/embedding_api/finetuned-model/chinese-roberta-nli-zh-all/"
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
