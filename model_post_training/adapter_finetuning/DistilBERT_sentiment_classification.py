# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DistilBERT_sentiment_classification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-24
# * Version     : 1.0.062414
# * Description : description
# * Link        : https://magazine.sebastianraschka.com/p/finetuning-llms-with-adapters
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
import time
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import torchmetrics
from datasets import load_dataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from data_provider.finetune.adapter.data_loader import IMDBDataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
os.environ['TOKENIZERS_PARALLELISM'] = "false"

from utils.log_util import logger


# ------------------------------
# dataset
# ------------------------------
imdb_dataset = load_dataset(
    "./dataset/finetune/adapter",
    data_files = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv",
    }
)
logger.info(f"imdb_dataset: \n{imdb_dataset}")

# ------------------------------
# tokenize dataset
# ------------------------------
# tokenzier
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
logger.info(f"Tokenizer input max length: {tokenizer.model_max_length}")
logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

# tokenize
def tokenize_text(batch):
    return tokenizer(
        batch["text"], 
        truncation=True, 
        padding=True
    )

imdb_tokenized = imdb_dataset.map(tokenize_text, batched = True, batch_size=None)
del imdb_dataset
imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------
# set up dataloaders
# ------------------------------
train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

# ------------------------------
# model
# ------------------------------
# model
model = AutoModelForSequenceClassification.from_pretrained(
    # "distilbert/distilbert-base-uncased", 
    "distilbert-base-uncased", 
    num_labels=2
)
logger.info(f"model architecture: \n{model}")

# ------------------------------
# 1.finetuning baseline: 
# finetuning the last layers of a DistilBERT model on a movie review dataset
# ------------------------------
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Unfreeze the two output layers
for param in model.pre_classifier.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logger.info(f"Total number of trainable parameters: {count_parameters(model)}")

# ------------------------------
# 2.finetuning with adapters
# add Adapter layers
# ------------------------------
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Add Adapter layers
def make_adapter(in_dim, bottleneck_dim, out_dim):
    adapter_layers = torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )
    return adapter_layers

total_size = 0
bottleneck_size = 32 # hyperparameter
for block_idx in range(6):
    ###################################################
    # insert 1st adapter layer into transformer block
    ###################################################
    orig_layer_1 = model.distilbert.transformer.layer[block_idx].attention.out_lin
    adapter_layers_1 = make_adapter(
        in_dim=orig_layer_1.out_features, 
        bottleneck_dim=bottleneck_size, 
        out_dim=orig_layer_1.out_features
    )
    new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
    model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1
    total_size += count_parameters(adapter_layers_1)
    ###################################################
    # insert 2nd adapter layer into transformer block
    ###################################################
    orig_layer_2 = model.distilbert.transformer.layer[block_idx].ffn.lin2
    adapter_layers_2 = make_adapter(
        in_dim=orig_layer_2.out_features, 
        bottleneck_dim=bottleneck_size, 
        out_dim=orig_layer_2.out_features
    )
    new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
    model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2
    
    total_size += count_parameters(adapter_layers_2)

logger.info("Number of adapter parameters added:", total_size)


# ------------------------------
# 3.finetuning all layers
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_param = count_parameters(model.pre_classifier) + count_parameters(model.classifier)
print("Parameters in last 2 layers:", num_param)


# ------------------------------
# 4.Insert adapter layers and finetuning all layers
# ------------------------------
def make_adapter(in_dim, bottleneck_dim, out_dim):
    adapter_layers = torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        torch.nn.GELU(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )
    return adapter_layers

total_size = 0
bottleneck_size = 32 # hyperparameter
for block_idx in range(6):
    ###################################################
    # insert 1st adapter layer into transformer block
    ###################################################
    orig_layer_1 = model.distilbert.transformer.layer[block_idx].attention.out_lin
    adapter_layers_1 = make_adapter(
        in_dim=orig_layer_1.out_features, 
        bottleneck_dim=bottleneck_size, 
        out_dim=orig_layer_1.out_features
    )
    new_1 = torch.nn.Sequential(orig_layer_1, *adapter_layers_1)
    model.distilbert.transformer.layer[block_idx].attention.out_lin = new_1
    total_size += count_parameters(adapter_layers_1)
    ###################################################
    # insert 2nd adapter layer into transformer block
    ###################################################
    orig_layer_2 = model.distilbert.transformer.layer[block_idx].ffn.lin2
    adapter_layers_2 = make_adapter(
        in_dim=orig_layer_2.out_features, 
        bottleneck_dim=bottleneck_size, 
        out_dim=orig_layer_2.out_features
    )
    new_2 = torch.nn.Sequential(orig_layer_2, *adapter_layers_2)
    model.distilbert.transformer.layer[block_idx].ffn.lin2 = new_2
    
    total_size += count_parameters(adapter_layers_2)

logger.info("Number of adapter parameters added:", total_size)


# ------------------------------
# model training
# ------------------------------
# model
class CustomLightningModule(L.LightningModule):
    
    def __init__(self, model, learning_rate=5e-5):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr = self.learning_rate,
        )

        return optimizer

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["label"]
        )
        self.log("train_loss", outputs["loss"])

        return outputs["loss"]
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["label"]
        )
        self.log("val_loss", outputs["loss"], prog_bar=True)

        logits = outputs["logits"]
        pred_labels = torch.argmax(logits, dim=1)
        self.val_acc(pred_labels, batch["label"])
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(
            batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["label"]
        )

        logits = outputs["logits"]
        pred_labels = torch.argmax(logits, dim=1)
        self.test_acc(pred_labels, batch["label"])
        self.log("accuracy", self.test_acc, prog_bar=True)

lightning_model = CustomLightningModule(model)

# callbacks
callbacks = [
    # save top 1 model
    ModelCheckpoint(
        save_top_k=1,
        mode="max",
        monitor="val_acc",
    )
]

# logger
lt_logger = CSVLogger(save_dir="./logs/", name=LOGGING_LABEL)

# trainer
trainer = L.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="gpu",
    precision="16-mixed",
    devices=[0],
    logger=lt_logger,
    log_every_n_steps=10,
)

# model training
start = time.time()
trainer.fit(
    model=lightning_model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
end = time.time()
elapsed = end - start
logger.info(f"Time elapsed: {elapsed/60:.2f}min")

# model testing
trainer.test(
    lightning_model,
    dataloaders=train_dataloader,
    ckpt_path="best"
)
trainer.test(
    lightning_model,
    dataloaders=val_dataloader,
    ckpt_path="best"
)
trainer.test(
    lightning_model,
    dataloaders=test_dataloader,
    ckpt_path="best"
)


# ------------------------------
# compare
# ------------------------------
def experiments_compare_bar(accuracy, training_time):
    pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
