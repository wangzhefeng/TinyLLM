# -*- coding: utf-8 -*-

# ***************************************************
# * File        : judge.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-20
# * Version     : 1.0.102000
# * Description : 统一的 LLM-as-a-judge 评分入口。
#               合并原 ollama_evaluate.py / openai_evaluate.py 的重复评分逻辑：
#                 - backend="ollama"：走 inference.api.ollama_api（本地 Ollama 服务）
#                 - backend="openai"：走 inference.api.openai_api（OpenAI API）
# * Link        : link
# * Requirement : tqdm；openai 后端需 pip install openai 及环境变量 OPENAI_API_KEY
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
# 仓根路径由本文件位置推导（inference/evaluation/judge.py -> 上两级），不依赖启动时的 cwd
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.append(ROOT)
from tqdm import tqdm


from layers.tokenizers.chat import format_input_alpaca
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _build_score_prompt(entry, json_key):
    """
    构造 judge 评分 prompt（ollama / openai 共用一份）
    """
    prompt = (
        f"Given the input `{format_input_alpaca(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry[json_key]}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
        f"Respond with the integer number only."
    )

    return prompt


def _make_ollama_scorer(
    inference_server="ollama",
    model="llama3",
    url="http://localhost:11434/api/chat",
    seed=123,
    num_ctx=2048,
):
    """
    ollama judge：评分前检查 ollama 进程是否在运行
    """
    from inference.api.ollama_api import check_if_running, query_model
    # check ollama running
    ollama_running = check_if_running(process_name=inference_server)
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding")
    logger.info(f"Ollama running: {ollama_running}")

    def scorer(prompt):
        return query_model(prompt=prompt, model=model, url=url, seed=seed, num_ctx=num_ctx)

    return scorer


def _make_openai_scorer(client=None):
    """
    openai judge：client 缺省时自动 create_client()（从环境变量 OPENAI_API_KEY 读 key）
    """
    from inference.api.openai_api import create_client, run_chatgpt
    if client is None:
        client = create_client()

    def scorer(prompt):
        return run_chatgpt(prompt, client)

    return scorer


def generate_model_scores(json_data, json_key, backend="ollama", **kwargs):
    """
    统一的 LLM judge 评分入口。

    Args:
        json_data: 指令数据列表（alpaca 格式，需含 output 与 json_key 字段）
        json_key: 待评分的模型回答字段名
        backend: "ollama" 或 "openai"
        **kwargs:
            backend="ollama" 时: inference_server/model/url/seed/num_ctx
            backend="openai" 时: client（缺省自动 create_client()）

    Returns:
        scores: 每条数据的整数评分（0-100）
    """
    if backend == "ollama":
        scorer = _make_ollama_scorer(**kwargs)
    elif backend == "openai":
        scorer = _make_openai_scorer(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'ollama' or 'openai'.")

    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
            continue
        # query judge model to generate score
        prompt = _build_score_prompt(entry, json_key)
        score = scorer(prompt)
        # print the procession
        logger.info(f"\nDataset response:")
        logger.info(f">>, {entry['output']}")
        logger.info(f"\nModel response:")
        logger.info(f">>, {entry[json_key]}")
        logger.info(f"\nScore:")
        logger.info(f">>, {score}")
        logger.info(f"\n-------------------------")
        # save scores
        try:
            scores.append(int(score))
        except ValueError:
            logger.info(f"Could not convert score: {score}")
            continue

    return scores




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
