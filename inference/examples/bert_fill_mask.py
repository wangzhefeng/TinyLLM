# -*- coding: utf-8 -*-

# ***************************************************
# * File        : bert_fill_mask.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101800
# * Description : BERT 推理示例：加载 run_bert.py 训练的 checkpoint，做 fill-mask（MLM）与句子对（NSP）推理
# * Link        : docs/40-models/bert.md
# * Requirement : torch
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
import re

import torch

from models.bert.model import Model
from utils.device import device_setting
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]

MASK_TOKEN = "[MASK]"


def args_parse():
    parser = argparse.ArgumentParser(description="BERT fill-mask / NSP inference demo")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/bert/bert_pretrain_best.pt", help="checkpoint from run_bert.py")
    parser.add_argument("--vocab", type=str, default="./dataset/finetuning/THUCNews/index_2_word.txt", help="vocab file (one token per line)")
    parser.add_argument("--text1", type=str, default="今天天气[MASK]好", help="sentence A, use [MASK] to mark positions to predict")
    parser.add_argument("--text2", type=str, default="我们决定去公园散步", help="sentence B (for NSP); empty string to skip")
    parser.add_argument("--topk", type=int, default=5, help="top-k predictions per [MASK]")
    return parser.parse_args()


def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        index_2_word = f.read().split("\n")
    word_2_index = {w: idx for idx, w in enumerate(index_2_word)}
    return word_2_index, index_2_word


def load_model(checkpoint: str, device):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Model(state["config"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded from {checkpoint} (epoch={state.get('epoch')}, val_loss={state.get('val_loss'):.4f})")
    return model, state["config"]


def tokenize(text: str, word_2_index):
    """
    字符级切分，保留 [MASK] 为独立 token；未登录字映射为 [UNK]
    """
    unk_idx = word_2_index["[UNK]"]
    tokens = []
    for part in re.split(r"(\[MASK\])", text):
        if part == MASK_TOKEN:
            tokens.append(word_2_index[MASK_TOKEN])
        else:
            tokens.extend(word_2_index.get(ch, unk_idx) for ch in part)
    return tokens


def build_inputs(text1: str, text2: str, word_2_index, device, max_len: int):
    """
    构造与预训练一致的输入：[CLS] text1 [SEP] text2 [SEP]，segment 0/1。
    模型内部已从 input_ids 推导 padding mask，无需手动 padding；
    仅需保证长度不超过训练时的 max_len（位置嵌入上限）。
    """
    text1_idx = tokenize(text1, word_2_index)[:62]
    text2_idx = tokenize(text2, word_2_index)[:63] if text2 else []
    input_ids = [word_2_index["[CLS]"]] + text1_idx + [word_2_index["[SEP]"]] + text2_idx + [word_2_index["[SEP]"]]
    seg_ids = [0] * (len(text1_idx) + 2) + [1] * (len(text2_idx) + 1)
    if len(input_ids) > max_len:
        raise ValueError(f"Input length {len(input_ids)} exceeds max_len {max_len}")
    batch_idx = torch.tensor([input_ids], device=device)
    batch_seg_idx = torch.tensor([seg_ids], device=device)
    mask_positions = [i for i, t in enumerate(input_ids) if t == word_2_index[MASK_TOKEN]]
    return batch_idx, batch_seg_idx, mask_positions


@torch.no_grad()
def fill_mask(model, batch_idx, batch_seg_idx, mask_positions, index_2_word, topk: int):
    """
    对每个 [MASK] 位置输出 top-k 预测 token 及概率
    """
    sequence_out, pooled = model.bert(batch_idx, batch_seg_idx)
    logits = model.cls_mask(sequence_out)  # (1, seq_len, vocab)
    results = []
    for pos in mask_positions:
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_probs, top_ids = probs.topk(topk)
        results.append([
            (index_2_word[tid], round(p.item(), 4))
            for p, tid in zip(top_probs, top_ids)
        ])
    return results


@torch.no_grad()
def predict_nsp(model, batch_idx, batch_seg_idx):
    """
    NSP：判断 text2 是否为 text1 的下一句
    """
    _, pooled = model.bert(batch_idx, batch_seg_idx)
    probs = torch.softmax(model.cls_next(pooled), dim=-1)[0]
    return {"is_next": round(probs[1].item(), 4), "not_next": round(probs[0].item(), 4)}



# 测试代码 main 函数
def main():
    args = args_parse()
    device = device_setting(verbose=True)
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}. Train first: python run_bert.py --max_samples 2000 --train_epochs 2")
        sys.exit(1)
    word_2_index, index_2_word = load_vocab(args.vocab)
    model, config = load_model(args.checkpoint, device)
    batch_idx, batch_seg_idx, mask_positions = build_inputs(args.text1, args.text2, word_2_index, device, config["max_position_embeddings"])
    # MLM
    if mask_positions:
        logger.info(f"text1: {args.text1}")
        for pos, preds in zip(mask_positions, fill_mask(model, batch_idx, batch_seg_idx, mask_positions, index_2_word, args.topk)):
            logger.info(f"[MASK] at pos {pos}: " + ", ".join(f"{tok}({p})" for tok, p in preds))
    else:
        logger.info("No [MASK] in text1, skip fill-mask.")
    # NSP
    if args.text2:
        logger.info(f"NSP('{args.text1}' -> '{args.text2}'): {predict_nsp(model, batch_idx, batch_seg_idx)}")

if __name__ == "__main__":
    main()
