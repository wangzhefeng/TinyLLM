# -*- coding: utf-8 -*-

# ***************************************************
# * File        : text2vec_riddle.py
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


import pandas as pd
from uniem.finetuner import FineTuner

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# data load and preprocess
# ------------------------------
df = pd.read_json(
    'https://raw.githubusercontent.com/wangyuxinwhy/uniem/main/examples/example_data/riddle.jsonl', 
    lines = True
)
# records = df.to_dict('records')
# print(records[0])
# print(records[1])

df = df.rename(columns = {
    'instruction': 'text', 
    'output': 'text_pos'
})

# ------------------------------
# model finetune
# ------------------------------
# 指定训练的模型为 m3e-small
finetuner = FineTuner.from_pretrained(
    'shibing624/text2vec-base-chinese-sentence', 
    dataset = df.to_dict('records')
)
fintuned_model = finetuner.run(
    epochs = 3, 
    output_dir = "E:/projects/llms_proj/llm_proj/embedding_api/finetuned-model/text2vec-riddle/" 
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
