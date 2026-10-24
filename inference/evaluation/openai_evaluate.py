# -*- coding: utf-8 -*-

# ***************************************************
# * File        : openai_evaluate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-18
# * Version     : 0.2.102000
# * Description : 【已合并到 inference/evaluation/judge.py，本文件仅为兼容保留】
#               保留原 generate_model_scores(json_data, json_key, client) 签名，
#               内部委托 judge.generate_model_scores(backend="openai")。
# * Link        : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/03_model-evaluation/llm-instruction-eval-openai.ipynb
# * Requirement : pip install openai；环境变量 OPENAI_API_KEY 中配置 API key
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
# 仓根路径由本文件位置推导（inference/evaluation/openai_evaluate.py -> 上两级），不依赖启动时的 cwd
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)


from inference.evaluation.judge import generate_model_scores as _judge_generate_model_scores

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def generate_model_scores(json_data, json_key, client):
    """
    generate model evaluate scores
    """
    return _judge_generate_model_scores(
        json_data,
        json_key,
        backend="openai",
        client=client,
    )




# 测试代码 main 函数
def main():
    import json
    from utils.load_save_data import load_json_data
    from inference.api.openai_api import create_client

    # instruction data path（数据集目录为 dataset/finetuning/；
    # instruction-data.json 由 post_training/sft 的 instruction_follow 数据下载脚本生成）
    data_path="./dataset/finetuning/instruction-data.json"

    # load instruction data
    json_data = load_json_data(data_path)

    # openai client
    client = create_client()
    
    for model in ("model 1 response", "model 2 response"):
        scores = generate_model_scores(json_data, model, client)
        print(f"\n{model}")
        print(f"Number of scores: {len(scores)} of {len(json_data)}")
        print(f"Average score: {sum(scores)/len(scores):.2f}\n")

        # Optionally save the scores
        save_dir = Path("scores")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"gpt4-{model.replace(' ', '-')}.json"
        with open(save_path, "w") as file:
            json.dump(scores, file)

if __name__ == "__main__":
    main()
