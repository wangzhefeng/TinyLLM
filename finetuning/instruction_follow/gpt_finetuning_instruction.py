# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_finetuning_instruction.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021622
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import re
import json
import time
from tqdm import tqdm

import tiktoken
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from finetuning.instruction_follow.data_load import load_file
from finetuning.instruction_follow.data_loader import create_dataloader
from models.gpt import Model
from pretrained_weights_load.openai_gpt2_weights_load_hf import load_weights
from utils.argsparser_tools import DotDict
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# create dataset and dataloader
# ------------------------------
# data
data = load_file(file_path = "./dataset/finetuning/instruction-data.json")

# data split
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.10)
valid_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion:train_portion+test_portion]
valid_data = data[train_portion+test_portion:]
# logger.info(f"Training data length: {len(train_data)}")
# logger.info(f"Test data length: {len(test_data)}")
# logger.info(f"Validation data length: {len(valid_data)}")

# dataset and dataloader
train_dataset, train_dataloader = create_dataloader(
    data = train_data,
    batch_size = 8,
    shuffle = True,
    drop_last = True,
    num_workers = 0,
)
test_dataset, test_dataloader = create_dataloader(
    data = test_data,
    batch_size = 8,
    shuffle = False,
    drop_last = False,
    num_workers = 0,
)
valid_dataset, valid_dataloader = create_dataloader(
    data = valid_data,
    batch_size = 8,
    shuffle = False,
    drop_last = False,
    num_workers = 0,
)
# test
logger.info(f"Train loader:")
for inputs, targets in train_dataloader:
    logger.info(f"inputs: \n{inputs[7]} \ninputs.shap: {inputs.shape}")
    logger.info(f"targets: \n{targets[7]} \ntargets.shape: {targets.shape}")
    break


# ------------------------------
# initializing a model with pretrained weights
# Loading pretrained LLM
# ------------------------------
choose_model = "gpt2-medium (355M)"
input_prompt = "Every effort moves you"

# model base cofig
base_config = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "dropout": 0.0,          # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}
# huggingface allowed model names
gpt2_hf_model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}
gpt2_model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
base_config.update(gpt2_model_configs[choose_model])
base_config = DotDict(base_config)

# huggingface gpt2 model
from transformers import GPT2Model
gpt2_hf = GPT2Model.from_pretrained(
    gpt2_hf_model_names[choose_model],
    cache_dir = "./downloaded_models/gpt2_model"
)
gpt2_hf.eval();

# custom gpt model
model = Model(base_config)
load_weights(model, gpt2_hf, base_config)
model.eval();

# test
torch.manual_seed(123)
from finetuning.instruction_follow.data_loader import format_input
input_text = format_input(valid_data[0])
logger.info(f"input text: \n{input_text}")
from layers.tokenization import text_to_token_ids, token_ids_to_text
from models.gpt_generate import generate
token_ids = generate(
    model = model, 
    token_idx = text_to_token_ids(input_text),
    max_new_tokens = 35,
    context_size = base_config.context_length,
    eos_id = 50256,
)
generated_text = token_ids_to_text(token_ids)
logger.info(f"generated text: \n{generated_text}")
response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
)
logger.info(f"response text: \n{response_text}")

# ------------------------------
# Finetuning LLM on instruction data
# ------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Logits of last output token
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


# Disable gradient tracking for efficiency because we are not training, yet
model.to(device)
torch.manual_seed(123)
with torch.no_grad(): 
    train_loss = calc_loss_loader(train_dataloader, model, device, num_batches=5)
    val_loss = calc_loss_loader(valid_dataloader, model, device, num_batches=5)
    # test_loss = calc_loss_loader(test_dataloader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
# print(f"Test loss: {test_loss:.3f}")



# model training
start_time = time.time()
torch.manual_seed(123)
num_epochs = 2
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")
train_losses, val_losses, tokens_seen = train_model_simple(
    model, 
    train_dataloader, 
    valid_dataloader, 
    optimizer, 
    device,
    num_epochs=num_epochs, 
    eval_freq=5, 
    eval_iter=5,
    start_context = format_input(valid_data[0]), 
    tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# ------------------------------
# Extracting and saving responses
# ------------------------------
torch.manual_seed(123)
for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=base_config.context_length,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")


for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text).to(device),
        max_new_tokens=256,
        context_size=base_config.context_length,
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    test_data[i]["model_response"] = response_text

result_path = "./saved_results/test_results/"
os.makedirs(result_path, exist_ok=True)
with open(os.path.join(result_path, "instruction-data-with-response.json"), "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing


# Save model
pretrained_model_path = "./saved_results/pretrained_models/"
file_name = os.path.join(pretrained_model_path, f"{re.sub(r'[ ()]', '', choose_model) }-sft.pth")
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model
# model.load_state_dict(torch.load(os.path.join(pretrained_model_path, "gpt2-medium355M-sft.pth")))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
