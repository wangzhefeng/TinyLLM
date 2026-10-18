# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101800
# * Description : BERT 情感分类：加载 run_bert.py 预训练 backbone + 分类头（支持冻结对照）
# * Link        : docs/40-models/bert.md
# * Requirement : torch
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from models.bert.model import Model
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def load_pretrain_checkpoint(checkpoint: str, device):
    """
    加载 run_bert.py 保存的 checkpoint，返回 (config, model_state_dict)
    """
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    logger.info(f"Pretrain checkpoint loaded from {checkpoint} "
                f"(epoch={state.get('epoch')}, val_loss={state.get('val_loss'):.4f})")
    return state["config"], state["model_state_dict"]


class BertSentimentClassifier(nn.Module):
    """
    从零 BERT（models.bert.model）+ 情感分类头。

    - init_mode="pretrained"：加载 run_bert.py 的 MLM+NSP checkpoint，迁移 encoder 表示；
    - init_mode="scratch"：同结构随机初始化（对照组，验证预训练的价值）；
    - freeze_backbone=True：只训练分类头（特征提取模式，第二种对照）。
    """

    def __init__(self, config, num_classes: int = 2, init_mode: str = "scratch",
                 checkpoint: str = None, freeze_backbone: bool = False, device=None):
        super().__init__()
        backbone = Model(config)
        if init_mode == "pretrained":
            _, state_dict = load_pretrain_checkpoint(checkpoint, device or torch.device("cpu"))
            backbone.load_state_dict(state_dict)
            logger.info("Backbone initialized from pretrain checkpoint")
        else:
            logger.info("Backbone randomly initialized (scratch)")
        # 只保留 encoder（BertModel），丢弃 MLM/NSP 预训练头
        self.bert = backbone.bert
        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False
            logger.info("Backbone frozen, only classifier head is trainable")
        self.classifier = nn.Linear(config["hidden_size"], num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, seg_ids, labels=None):
        # pooled_output: [CLS] 位置的聚合表示
        _, pooled_output = self.bert(input_ids, seg_ids)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss_fn(logits, labels)
        return logits

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
