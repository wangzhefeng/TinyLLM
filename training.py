# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-23
# * Version     : 1.0.012322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import time
import warnings
# from importlib.metadata import version
# logger.info(f"torch version: {version('torch')}")
# logger.info(f"tiktoken version: {version('tiktoken')}")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_provider.data_load_pretrain import data_load
from data_provider.data_loader import create_dataloader
from models.gpt import GPTModel, generate_text_simple
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"Using {device} device.")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
    # add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # logger.info(f"encoded tensor: {encoded_tensor}")
    # logger.info(f"encoded tensor shape: {encoded_tensor.shape}")
    
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # remove batch dimension
    flat = token_ids.squeeze(0)
    decoded_text = tokenizer.decode(flat.tolist())
    # logger.info(f"decoded text: {decoded_text}")
    
    return decoded_text


def _calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

    return loss


def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = _calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches


def train_model_simple(model, 
                       train_loader, 
                       val_loader, 
                       optimizer, 
                       device, 
                       num_epochs, 
                       eval_freq, 
                       eval_iter, 
                       start_context, 
                       tokenizer):
    # initialize list to track losses and tokens seen
    train_losses, val_losses = [], []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = -1
    # main training loop
    for epoch in range(num_epochs):
        # training mode
        model.train()
        for input_batch, target_batch in train_loader:
            # 前向传播
            optimizer.zero_grad()
            loss = _calc_loss_batch(input_batch, target_batch, model, device)
            # 后向传播
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1
            # optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"Ep {epoch + 1} (Step {global_step:06d}): Train loss: {train_loss:.3f}, Val loss {val_loss:.3f}")
        # print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, 
            idx = encoded,
            max_new_tokens = 50,
            context_size = context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    logger.info(decoded_text.replace('\n', ' '))
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize = (5, 3))
    # plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label = "Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle = "-.", label = "Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc = "upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))
    # create a second x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha = 0)
    ax2.set_xlabel("Tokens seen")

    # adjust layout to make room
    fig.tight_layout()
    # plt.savefig("loss_plot.pdf")
    plt.show()


start_time = time.time()
torch.manual_seed(123)
# params
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}
# model
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
num_epochs = 10

# data load
raw_text = data_load()

# dataset and dataloader
train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]
train_loader = create_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = create_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

# tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")

# model training
train_losses, val_losses, tokens_seen = train_model_simple(
    model, 
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs = num_epochs,
    eval_freq = 5,
    eval_iter = 5,
    start_context = "Every effort moves you",
    tokenizer = tokenizer,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
logger.info(f"Training Completed in {execution_time_minutes:.2f} minutes.")

# loss visual
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)




# 测试代码 main 函数
def main():
    """
    # params
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }
    # seed
    torch.manual_seed(123)
    
    # ------------------------------
    # generate text
    # ------------------------------
    # model
    model = GPTModel(GPT_CONFIG_124M)
    # disable dropout durning inference
    model.eval()
    
    # text
    start_context = "Every effort move you"
    
    # tokenizer 
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # token IDs
    token_ids = generate_text_simple(
        model = model,
        idx = text_to_token_ids(start_context, tokenizer),
        max_new_tokens = 10,
        context_size = GPT_CONFIG_124M["context_length"],
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids, tokenizer)}")

    # ------------------------------
    # text generation loss: cross-entropy and perplexity
    # ------------------------------
    # input data
    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                           [40,    1107, 588]])   #  "I really like"]
    logger.info(f"inputs shape: {inputs.shape}")
    
    # target data
    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                            [1107,  588, 11311]]) #  " really like chocolate"]
    logger.info(f"targets shape: {targets.shape}")
    
    # model inference
    with torch.no_grad():
        logits = model(inputs)
    logger.info(f"logits: \n{logits}")
    logger.info(f"logits shape: {logits.shape}")
    
    # softmax
    probas = torch.softmax(logits, dim=-1)
    logger.info(f"probas: \n{probas}")
    logger.info(f"probas.shape: {probas.shape}")
    
    # argmax
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    logger.info(f"Token IDs: \n{token_ids}")
    logger.info(f"Token IDs shape: {token_ids.shape}")
    logger.info(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    logger.info(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    
    # token probabilities corresponding to the target indices
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    logger.info(f"Text 1: \n{target_probas_1}")
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    logger.info(f"Text 2: \n{target_probas_2}")
    
    # compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat([target_probas_1, target_probas_2]))
    logger.info(f"log_probas: \n{log_probas}")
    avg_log_probas = torch.mean(log_probas)
    logger.info(f"avg log probas: {avg_log_probas}")
    neg_avg_log_probas = avg_log_probas * -1
    logger.info(f"neg_avg_log_probas: {neg_avg_log_probas}")
    # Logits have shape: (batch_size, num_tokens, vocab_size)
    logger.info(f"Logits shape: {logits.shape}")
    # Targets have shape: (batch_size, num_tokens)
    logger.info(f"Targets shape: {targets.shape}")
    
    # compute cross-entropy loss
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    logger.info(f"Flattened logtis: {logits_flat.shape}")
    logger.info(f"Flattened targets: {targets_flat.shape}")
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    logger.info(f"loss: {loss}")
    # perplexity
    perplexity = torch.exp(loss)
    logger.info(f"perplexity: {perplexity}")
    
    # 数据加载
    from tiny_model.TinyLLM.data_load import data_load
    raw_text = data_load()
    logger.info(f"raw_text[:99]: {raw_text[:99]}")
    logger.info(f"raw_text[:99]: {raw_text[-99:]}")

    # training and validation data loader create
    from tiny_model.TinyLLM.data_loader import create_dataloader
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]
    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    logger.info(f"Train loader:")
    for x, y in train_loader:
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")
    logger.info("Validation loader:")
    for x, y in val_loader:
        logger.info(f"x.shape: {x.shape}, y.shape: {y.shape}")
    
    # train, validation tokens
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()
    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()
    logger.info(f"Training tokens: {train_tokens}")
    logger.info(f"Validation tokens: {val_tokens}")
    logger.info(f"All tokens: {train_tokens + val_tokens}")

    # loss
    model.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    logger.info(f"Training loss: {train_loss}")
    logger.info(f"Validation loss: {val_loss}")
    """

if __name__ == "__main__":
    main()
