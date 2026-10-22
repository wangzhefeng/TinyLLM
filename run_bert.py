# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_bert.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-18
# * Version     : 1.0.101800
# * Description : BERT 从零预训练入口：MLM + NSP 双任务训练、验证与 checkpoint 保存
# * Link        : https://github.com/datawhalechina/hand-bert/blob/main/main.py
# * Requirement : torch, pandas, tqdm
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import argparse
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.bert.model import Model
from pre_training.data_provider.bert.data_loader import BertDataset
from utils.device import device_setting, torch_gc
from utils.log_util import logger
from utils.random_seed import set_seed, set_seed_worker

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def args_parse():
    parser = argparse.ArgumentParser(description="BERT from-scratch pretraining (MLM + NSP)")
    # task params
    parser.add_argument("--task_name", type=str, default="bert_pretrain", help="task name")
    parser.add_argument("--is_train", type=int, default=1, help="train flag")
    # data params
    parser.add_argument("--data_dir", type=str, default="./dataset/finetuning/THUCNews", help="data dir containing task2.csv and index_2_word.txt")
    parser.add_argument("--task_file", type=str, default="task2.csv", help="NSP sentence-pair dataset file")
    parser.add_argument("--vocab_file", type=str, default="index_2_word.txt", help="vocab file (one token per line)")
    parser.add_argument("--max_samples", type=int, default=-1, help="limit samples for quick runs, -1 means all")
    parser.add_argument("--dev_size", type=int, default=400, help="validation set size")
    parser.add_argument("--max_len", type=int, default=128, help="max sequence length (dataset truncates to 62+63+3)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
    # model params
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size")
    parser.add_argument("--feed_num", type=int, default=512, help="feed-forward inner size")
    parser.add_argument("--head_num", type=int, default=4, help="number of attention heads")
    parser.add_argument("--layer_num", type=int, default=2, help="number of encoder layers")
    parser.add_argument("--type_vocab_size", type=int, default=3, help="segment vocab size (0=text1, 1=text2, 2=padding)")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="embedding dropout")
    # train params
    parser.add_argument("--train_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="linear warmup ratio of total steps")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--log_interval", type=int, default=200, help="log every N steps")
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--checkpoints", type=str, default="./results/checkpoints/bert/", help="checkpoints dir")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume model weights from")

    return parser.parse_args()


def load_vocab(vocab_path: str):
    """
    读取 index_2_word.txt（每行一个 token），返回 word_2_index / index_2_word
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        index_2_word = f.read().split("\n")
    word_2_index = {w: idx for idx, w in enumerate(index_2_word)}
    logger.info(f"Vocab loaded from {vocab_path}, size = {len(word_2_index)}")
    return word_2_index, index_2_word


def load_task2_data(task_path: str, max_samples: int = -1):
    """
    读取 task2.csv，返回 (all_text1, all_text2, all_label)
    """
    df = pd.read_csv(task_path)
    if max_samples and max_samples > 0:
        df = df.iloc[:max_samples]
    logger.info(f"Task2 data loaded from {task_path}, samples = {len(df)}")
    return df["text1"].tolist(), df["text2"].tolist(), df["label"].tolist()


def build_config(args, vocab_size: int):
    """
    组装 models.bert.model.Model 所需的 config
    """
    return {
        "vocab_size": vocab_size,
        "hidden_size": args.hidden_size,
        "feed_num": args.feed_num,
        "head_num": args.head_num,
        "layer_num": args.layer_num,
        "max_position_embeddings": args.max_len,
        "type_vocab_size": args.type_vocab_size,
        "hidden_dropout_prob": args.hidden_dropout_prob,
    }


def build_dataloaders(all_text1, all_text2, all_label, word_2_index, args):
    """
    末尾 dev_size 条作为验证集，其余作为训练集
    """
    if args.dev_size >= len(all_label):
        raise ValueError(f"dev_size({args.dev_size}) >= dataset size({len(all_label)})")
    train_text1, train_text2, train_labels = all_text1[:-args.dev_size], all_text2[:-args.dev_size], all_label[:-args.dev_size]
    dev_text1, dev_text2, dev_labels = all_text1[-args.dev_size:], all_text2[-args.dev_size:], all_label[-args.dev_size:]
    logger.info(f"Train samples = {len(train_labels)}, dev samples = {len(dev_labels)}")

    train_dataset = BertDataset(train_text1, train_text2, train_labels, args.max_len, word_2_index)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=set_seed_worker,
        drop_last=True,
    )
    dev_dataset = BertDataset(dev_text1, dev_text2, dev_labels, args.max_len, word_2_index)
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=set_seed_worker,
    )
    return train_dataloader, dev_dataloader


def lr_lambda_factory(total_steps: int, warmup_ratio: float):
    """
    线性 warmup + 线性衰减
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return lr_lambda


def evaluate(model, dev_dataloader, device):
    """
    在 dev 集上计算联合 loss、MLM 准确率与 NSP 准确率
    """
    model.eval()
    total_loss, total_batches = 0.0, 0
    mask_correct, mask_total = 0, 0
    next_correct, next_total = 0, 0
    with torch.no_grad():
        for batch_idx, batch_label, batch_mask_val, batch_seg_idx in dev_dataloader:
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            batch_mask_val = batch_mask_val.to(device)
            batch_seg_idx = batch_seg_idx.to(device)
            # dev loss
            loss = model(batch_idx, batch_seg_idx, batch_mask_val, batch_label)
            total_loss += loss.item()
            total_batches += 1
            # dev accuracy
            pre_mask, pre_next = model(batch_idx, batch_seg_idx)
            mask_correct += (pre_mask[batch_mask_val != 0] == batch_mask_val[batch_mask_val != 0]).sum().item()
            mask_total += (batch_mask_val != 0).sum().item()
            next_correct += (pre_next == batch_label).sum().item()
            next_total += len(batch_label)
    avg_loss = total_loss / max(1, total_batches)
    acc_mask = mask_correct / max(1, mask_total)
    acc_next = next_correct / max(1, next_total)
    return avg_loss, acc_mask, acc_next


def save_checkpoint(model, optimizer, epoch, val_loss, config, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")


def train(args):
    # device
    device = device_setting(verbose=True)
    # vocab
    word_2_index, index_2_word = load_vocab(str(Path(args.data_dir) / args.vocab_file))
    # data
    all_text1, all_text2, all_label = load_task2_data(str(Path(args.data_dir) / args.task_file), args.max_samples)
    train_dataloader, dev_dataloader = build_dataloaders(all_text1, all_text2, all_label, word_2_index, args)
    # model
    config = build_config(args, vocab_size=len(word_2_index))
    model = Model(config).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Resumed model weights from {args.resume} (epoch={state.get('epoch')})")
    logger.info(f"Model config: {config}")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.train_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(total_steps, args.warmup_ratio))
    # early stopping（最优 dev loss 存 checkpoint，统一格式便于推断脚本加载）
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(args.checkpoints) / "bert_pretrain_best.pt")
    best_val_loss = float("inf")
    patience_counter = 0
    # training loop
    logger.info(f"{180 * '='}")
    logger.info(f"Start training: epochs={args.train_epochs}, steps/epoch={len(train_dataloader)}, total_steps={total_steps}")
    for epoch in range(args.train_epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_steps = 0.0, 0
        for step, (batch_idx, batch_label, batch_mask_val, batch_seg_idx) in enumerate(train_dataloader):
            batch_idx = batch_idx.to(device)
            batch_label = batch_label.to(device)
            batch_mask_val = batch_mask_val.to(device)
            batch_seg_idx = batch_seg_idx.to(device)
            # forward + loss
            loss = model(batch_idx, batch_seg_idx, batch_mask_val, batch_label)
            train_loss += loss.item()
            train_steps += 1
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # log
            if (step + 1) % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{args.train_epochs} | step {step+1}/{len(train_dataloader)} | "
                            f"train loss: {train_loss / train_steps:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")
        # validation
        val_loss, acc_mask, acc_next = evaluate(model, dev_dataloader, device)
        logger.info(f"Epoch {epoch+1}/{args.train_epochs} done in {time.time() - epoch_start:.1f}s | "
                    f"train loss: {train_loss / max(1, train_steps):.4f} | dev loss: {val_loss:.4f} | "
                    f"acc_mask: {acc_mask * 100:.3f}% | acc_next: {acc_next * 100:.3f}%")
        # early stopping：dev loss 改善则保存最优 checkpoint，否则累计 patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, config, ckpt_path)
        else:
            patience_counter += 1
            logger.info(f"EarlyStopping counter: {patience_counter} out of {args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    # final save
    final_path = str(Path(args.checkpoints) / "bert_pretrain_final.pt")
    save_checkpoint(model, optimizer, args.train_epochs, val_loss, config, final_path)
    logger.info(f"Training finished. Best checkpoint: {ckpt_path}, final checkpoint: {final_path}")
    # empty cache
    torch_gc()
    return model



# 测试代码 main 函数
def main():
    # 参数解析
    args = args_parse()
    # 设置随机数
    set_seed(seed=args.seed)
    # 模型训练
    if args.is_train:
        logger.info(f">>>>>>>training: {args.task_name}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        train(args)

if __name__ == "__main__":
    main()
