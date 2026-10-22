# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101800
# * Description : BERT 情感分类微调入口：预训练 backbone + 分类头，
#                 支持 pretrained / scratch / freeze 三种模式对照
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
import time

import torch
from torch.utils.data import DataLoader

from post_training.sft.bert_sentiment.data import SentimentDataset, load_sentiment_data
from post_training.sft.bert_sentiment.model import BertSentimentClassifier, load_pretrain_checkpoint
from run_bert import load_vocab
from utils.device import device_setting
from utils.log_util import logger
from utils.random_seed import set_seed

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    parser = argparse.ArgumentParser(description="BERT sentiment finetuning (from-scratch BERT backbone)")
    # task params
    parser.add_argument("--task_name", type=str, default="bert_sentiment", help="task name")
    # data params
    parser.add_argument("--data_dir", type=str, default="./dataset/finetuning/sentiment", help="sentiment data dir")
    parser.add_argument("--vocab_file", type=str, default="./dataset/finetuning/THUCNews/index_2_word.txt", help="vocab file shared with pretraining")
    parser.add_argument("--max_samples", type=int, default=-1, help="limit train samples for quick runs, -1 means all")
    parser.add_argument("--max_len", type=int, default=128, help="max sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
    # model params
    parser.add_argument("--init_mode", type=str, default="pretrained", choices=["pretrained", "scratch"], help="backbone init: pretrained checkpoint or scratch")
    parser.add_argument("--checkpoint", type=str, default="./results/checkpoints/bert/bert_pretrain_best.pt", help="pretrain checkpoint (init_mode=pretrained)")
    parser.add_argument("--freeze_backbone", action="store_true", help="freeze backbone, train classifier head only")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    # scratch 模式下的模型结构（pretrained 模式从 checkpoint 读取 config）
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size (scratch mode)")
    parser.add_argument("--feed_num", type=int, default=512, help="feed-forward inner size (scratch mode)")
    parser.add_argument("--head_num", type=int, default=4, help="number of heads (scratch mode)")
    parser.add_argument("--layer_num", type=int, default=2, help="number of layers (scratch mode)")
    # train params
    parser.add_argument("--train_epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience (on dev accuracy)")
    parser.add_argument("--log_interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--checkpoints", type=str, default="./results/checkpoints/bert_sentiment/", help="checkpoints dir")

    return parser.parse_args()


def build_dataloaders(data_dir, word_2_index, args):
    (train_texts, train_labels), (valid_texts, valid_labels), (test_texts, test_labels) = load_sentiment_data(
        data_dir, args.max_samples
    )
    logger.info(f"Train/valid/test: {len(train_labels)}/{len(valid_labels)}/{len(test_labels)}")
    train_loader = DataLoader(
        SentimentDataset(train_texts, train_labels, word_2_index, args.max_len),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        SentimentDataset(valid_texts, valid_labels, word_2_index, args.max_len),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        SentimentDataset(test_texts, test_labels, word_2_index, args.max_len),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    return train_loader, valid_loader, test_loader


def build_model(args, word_2_index, device):
    if args.init_mode == "pretrained":
        config, _ = load_pretrain_checkpoint(args.checkpoint, device)
    else:
        config = {
            "vocab_size": len(word_2_index),
            "hidden_size": args.hidden_size,
            "feed_num": args.feed_num,
            "head_num": args.head_num,
            "layer_num": args.layer_num,
            "max_position_embeddings": args.max_len,
            "type_vocab_size": 3,
            "hidden_dropout_prob": 0.1,
        }
    model = BertSentimentClassifier(
        config,
        num_classes=args.num_classes,
        init_mode=args.init_mode,
        checkpoint=args.checkpoint if args.init_mode == "pretrained" else None,
        freeze_backbone=args.freeze_backbone,
        device=device,
    ).to(device)
    total, trainable = model.num_params()
    logger.info(f"Model params: total {total / 1e6:.2f}M, trainable {trainable / 1e6:.2f}M")
    return model


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    for input_ids, seg_ids, labels in dataloader:
        input_ids, seg_ids, labels = input_ids.to(device), seg_ids.to(device), labels.to(device)
        total_loss += model(input_ids, seg_ids, labels).item()
        preds = model(input_ids, seg_ids).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / max(1, len(dataloader)), correct / max(1, total)


def train(args):
    # device / seed / vocab
    device = device_setting(verbose=True)
    set_seed(seed=args.seed)
    word_2_index, _ = load_vocab(args.vocab_file)
    # data
    train_loader, valid_loader, test_loader = build_dataloaders(args.data_dir, word_2_index, args)
    # model
    model = build_model(args, word_2_index, device)
    # optimizer：只优化可训练参数（freeze 模式下只有分类头）
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    # train loop
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(args.checkpoints) / "bert_sentiment_best.pt")
    best_acc, patience_counter = 0.0, 0
    logger.info(f"{180 * '='}")
    logger.info(f"Start finetuning: init_mode={args.init_mode}, freeze={args.freeze_backbone}, "
                f"epochs={args.train_epochs}, steps/epoch={len(train_loader)}")
    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_steps = 0.0, 0
        for step, (input_ids, seg_ids, labels) in enumerate(train_loader):
            input_ids, seg_ids, labels = input_ids.to(device), seg_ids.to(device), labels.to(device)
            loss = model(input_ids, seg_ids, labels)
            train_loss += loss.item()
            train_steps += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.train_epochs} | step {step+1}/{len(train_loader)} | "
                            f"train loss: {train_loss / train_steps:.4f}")
        train_loss /= max(1, train_steps)
        valid_loss, valid_acc = evaluate(model, valid_loader, device)
        logger.info(f"Epoch {epoch+1}/{args.train_epochs} done in {time.time() - epoch_start:.1f}s | "
                    f"train loss: {train_loss:.4f} | dev loss: {valid_loss:.4f} | dev acc: {valid_acc * 100:.2f}%")
        # early stopping on dev accuracy
        if valid_acc > best_acc:
            best_acc = valid_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "dev_acc": best_acc,
                "init_mode": args.init_mode,
                "freeze_backbone": args.freeze_backbone,
                "model_state_dict": model.state_dict(),
            }, ckpt_path)
            logger.info(f"Best checkpoint saved to {ckpt_path} (dev acc: {best_acc * 100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    # test with best checkpoint
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(f"Training finished. Best dev acc: {best_acc * 100:.2f}% | test acc: {test_acc * 100:.2f}%")
    return best_acc, test_acc




# 测试代码 main 函数
def main():
    args = args_parse()
    train(args)

if __name__ == "__main__":
    main()
