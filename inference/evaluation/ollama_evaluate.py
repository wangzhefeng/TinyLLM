# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ollama_evaluate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-18
# * Version     : 0.2.102000
# * Description : 【已合并到 inference/evaluation/judge.py，本文件仅为兼容保留】
#               保留原 generate_model_scores 函数名与签名（下游
#               post_training/sft/exp/exp_finetune_gpt_instruction_flow_evaluate.py
#               依赖此签名），内部委托 judge.generate_model_scores(backend="ollama")。
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


from inference.evaluation.judge import generate_model_scores as _judge_generate_model_scores

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def generate_model_scores(
    json_data, 
    json_key, 
    inference_server,
    model = "llama3", 
    url = "http://localhost:11434/api/chat", 
    seed=123, 
    num_ctx=2048
    ):
    """
    generate model evaluate scores
    """
    return _judge_generate_model_scores(
        json_data,
        json_key,
        backend="ollama",
        inference_server=inference_server,
        model=model,
        url=url,
        seed=seed,
        num_ctx=num_ctx,
    )




# 测试代码 main 函数
def main(): 
    pass

if __name__ == "__main__":
    main()
