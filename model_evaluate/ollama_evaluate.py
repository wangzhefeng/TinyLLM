# -*- coding: utf-8 -*-

# ***************************************************
# * File        : evaluating_instruction_flow.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-18
# * Version     : 0.1.021800
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
from tqdm import tqdm


from data_provider.finetune.instruction_format import format_input_alpaca
from utils.inference_utils.ollama_api import check_if_running, query_model
from utils.log_util import logger

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
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            # query model to generate score
            prompt = (
                f"Given the input `{format_input_alpaca(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            # check ollama running
            ollama_running = check_if_running(process_name=inference_server)
            # inference
            if ollama_running:
                logger.info(f"Ollama running: {ollama_running}")
                # query model
                score = query_model(
                    prompt=prompt, 
                    model=model, 
                    url=url, 
                    seed=seed, 
                    num_ctx=num_ctx
                )
                # print the procession
                logger.info(f"\nDataset response:")
                logger.info(f">>, {entry['output']}")
                logger.info(f"\nModel response:")
                logger.info(f">>, {entry['model_response']}")
                logger.info(f"\nScore:")
                logger.info(f">>, {score}")
                logger.info(f"\n-------------------------")
            else:
                raise RuntimeError("Ollama not running. Launch ollama before proceeding") 

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
