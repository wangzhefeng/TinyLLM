# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_finetune_gpt_instruction_flow_evaluate.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-15
# * Version     : 1.0.031520
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)


from data_provider.load_save_data import load_json_data
from model_evaluate.ollama_evaluate import generate_model_scores
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelFinetuningInstructionFlowEvaluate:
    
    def __init__(self, args):
        self.args = args
    
    def _build(self):
        """
        build dataset
        """
        # data
        data = load_json_data(data_path = self.args.data_path)
        # data split
        train_portion = int(len(data) * self.args.train_ratio)
        test_portion = int(len(data) * self.args.test_ratio)
        valid_portion = len(data) - train_portion - test_portion
        train_data = data[:train_portion]
        test_data = data[train_portion:train_portion + test_portion]
        valid_data = data[train_portion + test_portion:]
        logger.info(f"Training data length: {len(train_data)}")
        logger.info(f"Test data length: {len(test_data)}")
        logger.info(f"Validation data length: {len(valid_data)}")

        return data, train_data, valid_data, test_data
    
    def evaluate(self):
        """
        evaluate finetuned model using another larger LLM
        """
        # load data
        data, train_data, valid_data, test_data = self._build()
        # evaluate
        scores = generate_model_scores(
            json_data = data, 
            json_key = "model_response", 
            inference_server = self.args.inference_server,
            model = self.args.inference_model,
            url = self.args.inference_server_url,
            seed = self.args.seed,
            num_ctx = self.args.num_ctx,
        )
        logger.info(f"Number of scores: {len(scores)} of {len(data)}")
        logger.info(f"Average score: {sum(scores) / len(scores):.2f}\n")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
