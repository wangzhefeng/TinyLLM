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
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tiktoken
import torch
import torch.nn as nn
from transformers import GPT2Model

# data
from data_provider.finetune.text_classification.data_loader import create_dataloader
# model
from models.gpt import Model
from model_train.gpt_generate import generate
# tokenizer
from tokenizer.tokenization import text_to_token_ids, token_ids_to_text
# other model
from models_load.openai_gpt2_models import gpt2_model_configs, gpt2_huggingface_models
from models_load.openai_gpt2_weights_load_hf import load_weights
# training
from model_train.calc_loss import _calc_loss_batch, _calc_loss_loader, _calc_loss
from model_train.calc_accuracy import _calc_accuracy_loader, _calc_accuracy
from model_train.train_funcs import _select_optimizer, _select_criterion
from model_train.plot_losses import plot_values_classifier
from model_train.save_load_model import _save_model
# utils
from utils.argsparser_tools import DotDict
from utils.device import device
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ModelFinetuningClassifier:

    def __init__(self, args):
        super(ModelFinetuningClassifier, self).__init__()
        self.args = args

    def _build_data(self):
        """
        create dataset and dataloader
        """
        # dataset and dataloader
        train_dataset, train_loader = create_dataloader(
            data_path = os.path.join(self.args.data_source, "train.csv"),
            max_length = None,
            batch_size = self.args.batch_size,
            shuffle = True,
            drop_last = True,
        )
        valid_dataset, valid_loader = create_dataloader(
            data_path = os.path.join(self.args.data_source, "valid.csv"),
            max_length = train_dataset.max_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
        )
        test_dataset, test_loader = create_dataloader(
            data_path = os.path.join(self.args.data_source, "test.csv"),
            max_length = train_dataset.max_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
        )

        return train_loader, valid_loader, test_loader

    def _build_model(self):
        """
        initializing a model with pretrained weights: gpt2-small (124M)
        TODO:
            cache_dir = "./downloaded_models/gpt2_model"
        """
        # model base config
        self.base_config = {
            "vocab_size": self.args.vocab_size,         # Vocabulary size: 50257
            "context_length": self.args.context_length, # Context length: 1024
            "dropout": self.args.dropout,               # Dropout rate: 0.0
            "qkv_bias": self.args.qkv_bias,             # Query-key-value bias: True
        }
        self.base_config.update(gpt2_model_configs[self.args.choose_model])
        self.base_config = DotDict(self.base_config)

        # assert train_dataset.max_length <= base_config["context_length"], (
        #     f"Dataset length {train_dataset.max_length} exceeds model's context "
        #     f"length {base_config['context_length']}. Reinitialize data sets with "
        #     f"`max_length={base_config['context_length']}`"
        # ) 
        gpt2_hf = GPT2Model.from_pretrained(
            gpt2_huggingface_models[self.args.choose_model],
            cache_dir = self.args.pretrained_model_path,
        )
        gpt2_hf.eval()

        # custom gpt model
        self.model = Model(self.base_config)
        load_weights(self.model, gpt2_hf, self.base_config)
        self.model.eval()

    def _finetune_model(self):
        """
        add a classification head
        """
        # model architecture
        # logger.info(f"model: {self.model}")

        # freeze model(make all layers non-trainable)
        for param in self.model.parameters():
            param.requires_grad = False

        # replace output layer
        self.model.out_head = nn.Linear(
            in_features = self.base_config.emb_dim, 
            out_features = self.args.num_classes
        )

        # make the last transformer block and final LayerNorm module 
        # connecting the last transformer block to the output layer trainable
        for param in self.model.trf_blocks[-1].parameters():
            param.requires_grad = True

        for param in self.model.final_norm.parameters():
            param.requires_grad = True

        # move model to device
        self.model.to(device)

    # ------------------------------
    # finetuning the model on supervised data
    # ------------------------------
    def evaluate_model(model, train_loader, val_loader, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = _calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = _calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()

        return train_loss, val_loss


    def _train_classifier_simple(self,
                                 model, 
                                 train_loader, 
                                 val_loader, 
                                 optimizer, 
                                 device, 
                                 train_epochs,
                                 eval_freq, 
                                 eval_iter):
        # Initialize lists to track losses and examples seen
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(train_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = _calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                examples_seen += input_batch.shape[0] # New: track examples instead of tokens
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Calculate accuracy after each epoch
            train_accuracy = _calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
            val_accuracy = _calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
            print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy*100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

        return train_losses, val_losses, train_accs, val_accs, examples_seen


    def train(self, model, train_epochs: int, train_loader, valid_loader, test_loader, device):
        """
        model training
        """
        start_time = time.time()
        optimizer = _select_optimizer(model)
        train_losses, \
        val_losses, \
        train_accs, \
        val_accs, \
        examples_seen = self._train_classifier_simple(
            model, 
            train_loader, 
            valid_loader, 
            optimizer, 
            device,
            train_epochs=train_epochs, 
            eval_freq=50, 
            eval_iter=5,
        )
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
        
        # training loss plot
        epochs_tensor = torch.linspace(0, train_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values_classifier(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        # training accuracy plot
        epochs_tensor = torch.linspace(0, train_epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values_classifier(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

        # accuracy
        train_accuracy = _calc_accuracy_loader(train_loader, model, device)
        val_accuracy = _calc_accuracy_loader(valid_loader, model, device)
        test_accuracy = _calc_accuracy_loader(test_loader, model, device)
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")


    # ------------------------------
    # using the LLM as a spam classifier
    # ------------------------------
    def classify_review(self, text, model, tokenizer, device, max_length=None, pad_token_id=50256):
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


    def inference(self, model, train_dataset):
        tokenizer = tiktoken.get_encoding("gpt2")

        # usage 1
        text_1 = (
            "You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award."
        )
        logger.info(self.classify_review(
            text_1, model, tokenizer, device, max_length=train_dataset.max_length
        ))

        # usage 2
        text_2 = (
            "Hey, just wanted to check if we're still on"
            " for dinner tonight? Let me know!"
        )

        print(self.classify_review(
            text_2, model, tokenizer, device, max_length=train_dataset.max_length
        ))




# 测试代码 main 函数
def main():
    # ------------------------------
    # params
    # ------------------------------
    data_path = os.path.join(ROOT, r"dataset\finetuning\sms_spam_collection")
    batch_size = 8
    num_classes = 2
    # ------------------------------
    # set seed
    # ------------------------------
    torch.manual_seed(123)
    # ------------------------------
    # data
    # ------------------------------
    train_loader, valid_loader, test_loader = _build_data(data_path, batch_size)
    # ------------------------------
    # model
    # ------------------------------
    model, base_config, choose_model = _build_model()
    # ------------------------------
    # model inference before finetuning
    # ------------------------------
    input_prompt = "Every effort moves you"
    token_ids = generate(
        model = model,
        token_idx = text_to_token_ids(input_prompt),
        max_new_tokens = 15,
        context_size = base_config.context_length,
    )
    logger.info(f"Output text: \n{token_ids_to_text(token_ids)}")

    # before finetune model as classifier
    text = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate(
        model = model,
        token_idx = text_to_token_ids(text),
        max_new_tokens = 23,
        context_size = base_config.context_length,
    )
    logger.info(f"Output text: {token_ids_to_text(token_ids)}")
    # ------------------------------
    # finetune model
    # ------------------------------
    model = _finetune_model(model, base_config, num_classes)
    # ------------------------------
    # calculate accuracy and loss before finetuning
    # ------------------------------
    # calculating the classification accuracy
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
    
    # calculating the classification accuracy
    _calc_accuracy(train_loader, valid_loader, test_loader, model, device)

    # calculating the classification loss
    _calc_loss(model, train_loader, valid_loader, test_loader, device)
    # ------------------------------
    # finetune model training
    # ------------------------------
    # optimizer
    optimizer = _select_optimizer(model)

    # criterion
    criterion = _select_criterion()

    # train model

if __name__ == "__main__":
    main()
