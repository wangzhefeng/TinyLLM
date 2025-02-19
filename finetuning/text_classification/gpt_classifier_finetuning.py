# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_finetuning_classifier.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021612
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
import time
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from finetuning.text_classification.data_loader_finetuning import create_dataloader
from models.gpt import Model
from models.gpt_generate import generate
from layers.tokenization import text_to_token_ids, token_ids_to_text
from pretrained_weights_load.gtp_download import download_and_load_gpt2
from pretrained_weights_load.openai_gpt2_weights_load_hf import load_weights
from utils.argsparser_tools import DotDict
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# create dataset and dataloader
# ------------------------------
# params
extracted_path = os.path.join(ROOT, r"dataset\finetuning\sms_spam_collection")
batch_size = 8

# dataset and dataloader
train_dataset, train_loader = create_dataloader(
    data_path = os.path.join(extracted_path, "train.csv"),
    max_length = None,
    batch_size = batch_size,
    shuffle = True,
    drop_last = True,
)
valid_dataset, valid_loader = create_dataloader(
    data_path = os.path.join(extracted_path, "valid.csv"),
    max_length = train_dataset.max_length,
    batch_size = batch_size,
    shuffle = False,
    drop_last = False,
)
test_dataset, test_loader = create_dataloader(
    data_path = os.path.join(extracted_path, "test.csv"),
    max_length = train_dataset.max_length,
    batch_size = batch_size,
    shuffle = False,
    drop_last = False,
)
logger.info(f"train_dataset.max_length: {train_dataset.max_length}")
logger.info(f"valid_dataset.max_length: {valid_dataset.max_length}")
logger.info(f"test_dataset.max_length: {test_dataset.max_length}")

# dataloader test
logger.info(f"Train loader:")
for input_batch, target_batch in train_loader:
    pass
logger.info(f"Input batch dim: {input_batch.shape}")
logger.info(f"Target batch dim: {target_batch.shape}")

logger.info(f"{len(train_loader)} training batches")
logger.info(f"{len(valid_loader)} validation batches")
logger.info(f"{len(test_loader)} test batches")

# ------------------------------
# initializing a model with pretrained weights
# ------------------------------
choose_model = "gpt2-small (124M)"
input_prompt = "Every effort moves you"

# model base cofig
base_config = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "dropout": 0.0,        # Dropout rate
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

assert train_dataset.max_length <= base_config["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {base_config['context_length']}. Reinitialize data sets with "
    f"`max_length={base_config['context_length']}`"
)

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


"""
# ------------------------------
# model inference
# ------------------------------
token_ids = generate(
    model = model,
    token_idx = text_to_token_ids(input_prompt),
    max_new_tokens = 15,
    context_size = base_config.context_length,
)
logger.info(f"Output text: {token_ids_to_text(token_ids)}")

# ------------------------------
# before finetune model as classifier
# ------------------------------
text = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

token_ids = generate(
    model=model,
    token_idx=text_to_token_ids(text),
    max_new_tokens=23,
    context_size=base_config.context_length,
)
logger.info(f"Output text: {token_ids_to_text(token_ids)}")
"""

# ------------------------------
# add a classification head
# ------------------------------
# model architecture
logger.info(f"model: {model}")

# freeze model(make all layers non-trainable)
for param in model.parameters():
    param.requires_grad = False

# replace output layer
torch.manual_seed(123)
num_classes = 2
model.out_head = nn.Linear(
    in_features = base_config.emb_dim, 
    out_features = num_classes,
)

# make the last transformer block and final LayerNorm module 
# connecting the last transformer block to the output layer trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# ------------------------------
# calculating the classification loss and accuracy
# ------------------------------
"""
# test
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
logger.info(f"inputs: \n{inputs} \ninputs shape: {inputs.shape}")
with torch.no_grad():
    outputs = model(inputs)
logger.info(f"outputs: \n{outputs} \noutputs shape: {outputs.shape}")

last_output_token = outputs[:, -1, :]
logger.info(f"Last output token: {outputs[:, -1, :]}")

probas = torch.softmax(outputs[:, -1, :], dim = -1)
logger.info(f"Last output token probas: {probas}")

label = torch.argmax(probas)
logger.info(f"Class label: {label.item()}")
"""

def calc_accuracy_loader(dataloader, model, device, num_batches = None):
    # model eval
    model.eval()
    # correct predictions and number of examples
    correct_preds, num_examples = 0, 0
    # number of batches
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    # calculate accuracy
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            pred_labels = torch.argmax(logits, dim=-1)
            num_examples += pred_labels.shape[0]
            correct_preds += (pred_labels == target_batch).sum().item()
        else:
            break
    
    return correct_preds / num_examples


model.to(device)
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
valid_accuracy = calc_accuracy_loader(valid_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

logger.info(f"Train accuracy: {train_accuracy * 100:.2f}%")
logger.info(f"Valid accuracy: {valid_accuracy * 100:.2f}%")
logger.info(f"Test accuracy: {test_accuracy * 100:.2f}%")


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


# Disable gradient tracking for efficiency because we are not training, yet
with torch.no_grad(): 
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(valid_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

# ------------------------------
# finetuning the model on supervised data
# ------------------------------
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()

    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen



# model training
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, \
val_losses, \
train_accs, \
val_accs, \
examples_seen = train_classifier_simple(
    model, 
    train_loader, 
    valid_loader, 
    optimizer, 
    device,
    num_epochs=num_epochs, 
    eval_freq=50, 
    eval_iter=5,
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()  # Adjust layout to make room
    # plt.savefig(f"{label}-plot.pdf")
    plt.show()

# training loss plot
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# training accuracy plot
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

# accuracy
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(valid_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# ------------------------------
# using the LLM as a spam classifier
# ------------------------------
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# usage 1
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
logger.info(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# usage 2
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

# model save
torch.save(
    model.state_dict(), 
    "./saved_results/pretrained_models/review_classifier.pth"
)

# model load
model_state_dict = torch.load(
    "./saved_results/pretrained_models/review_classifier.pth", 
    map_location=device, 
    weights_only=True
)
model.load_state_dict(model_state_dict)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
