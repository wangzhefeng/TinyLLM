# -*- coding: utf-8 -*-

# ***************************************************
# * File        : uniem_finetune.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-21
# * Version     : 0.1.102120
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


# finetune
dataset = load_dataset('shibing624/nli_zh', 'STS-B', cache_dir = 'cache')

finetuner = FineTuner.from_pretrained('moka-ai/m3e-small', dataset = dataset)
finetuned_model = finetuner.run(
    epochs = 3, 
    batch_size = 64, 
    lr = 3e-5, 
    output_dir = "E:/projects/llms_proj/llm_proj/embedding_api/finetuned-model/m3e-nli_zh/"
)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
