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
        self.train_dataset, train_loader = create_dataloader(
            data_path = os.path.join(self.args.data_source, "train.csv"),
            max_length = None,
            batch_size = self.args.batch_size,
            shuffle = True,
            drop_last = True,
        )
        valid_dataset, valid_loader = create_dataloader(
            data_path = os.path.join(self.args.data_source, "valid.csv"),
            max_length = self.train_dataset.max_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
        )
        test_dataset, test_loader = create_dataloader(
            data_path = os.path.join(self.args.data_source, "test.csv"),
            max_length = self.train_dataset.max_length,
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
        model = Model(self.base_config)
        load_weights(model, gpt2_hf, self.base_config)
        model.eval()
        
        return model

    def _finetune_model(self):
        """
        add a classification head
        """
        # build model and print model architecture
        self.model = self._build_model()
        logger.info(f"model: {self.model}")

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

    def _choose_tokenizer(self, tokenizer_model: str = "gpt2"):
        """
        choose tokenizer
        """
        tokenizer = tiktoken.get_encoding(tokenizer_model)

        return tokenizer
    # ------------------------------
    # finetuning the model on supervised data
    # ------------------------------
    def _evaluate_model(self, train_loader, val_loader, eval_iter):
        """
        evaluate model
        """
        # eval mode
        self.model.eval()
        # calculate loss
        with torch.no_grad():
            train_loss = _calc_loss_loader(train_loader, self.model, self.device, num_batches=eval_iter)
            val_loss = _calc_loss_loader(val_loader, self.model, self.device, num_batches=eval_iter)
        # train mode
        self.model.train()

        return train_loss, val_loss

    def train(self, train_epochs: int, eval_freq: int = 50, eval_iter: int = 5):
        """
        model training
        """
        # data loader
        train_loader, valid_loader, test_loader = self._build_data()
        # model
        self._finetune_model() 
        # optimizer
        optimizer = _select_optimizer(self.model)

        # record training start time
        training_start_time = time.time()

        # model training
        # Initialize lists to track losses and examples seen
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        examples_seen = 0
        global_step = -1
        # Main training loop
        for epoch in range(self.args.train_epochs):
            # Set model to training mode
            self.model.train()
            # batch training
            for input_batch, target_batch in train_loader:
                # Reset loss gradients from previous batch iteration
                self.optimizer.zero_grad()
                loss = _calc_loss_batch(input_batch, target_batch, self.model, self.device)
                # Calculate loss gradients
                loss.backward()
                # Update model weights using loss gradients
                self.optimizer.step()
                # New: track examples instead of tokens
                examples_seen += input_batch.shape[0]
                global_step += 1
                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(train_loader, valid_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    logger.info(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Calculate accuracy after each epoch
            train_accuracy = _calc_accuracy_loader(train_loader, self.model, device, num_batches=eval_iter)
            val_accuracy = _calc_accuracy_loader(valid_loader, self.model, device, num_batches=eval_iter)
            logger.info(f"Training accuracy: {train_accuracy*100:.2f}% | ")
            logger.info(f"Validation accuracy: {val_accuracy*100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
        
        # training end time
        training_end_time = time.time()
        execution_time_minutes = (training_end_time - training_start_time) / 60
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
        train_accuracy = _calc_accuracy_loader(train_loader, self.model, device)
        val_accuracy = _calc_accuracy_loader(valid_loader, self.model, device)
        test_accuracy = _calc_accuracy_loader(test_loader, self.model, device)
        logger.info(f"Training accuracy: {train_accuracy*100:.2f}%")
        logger.info(f"Validation accuracy: {val_accuracy*100:.2f}%")
        logger.info(f"Test accuracy: {test_accuracy*100:.2f}%")

    def inference(self, text: str):
        """
        using the LLM as a spam classifier
        """
        # tokenizer
        tokenizer = self._choose_tokenizer()
        # eval mode
        self.model.eval()
        # Prepare inputs to the model
        input_ids = tokenizer.encode(text)
        # Truncate sequences if they too long
        supported_context_length = self.model.pos_emb.weight.shape[0]
        input_ids = input_ids[:min(self.train_dataset.max_length, supported_context_length)]
        # Pad sequences to the longest sequence
        input_ids += [self.train_dataset.pad_token_id] * (self.train_dataset.max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device = self.device).unsqueeze(0)  # add batch dimension
        # Model inference
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]  # Logits of the last output token
        # probability
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return the classified result
        return "spam" if predicted_label == 1 else "not spam"




# 测试代码 main 函数
def main():
    # usage 1
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    # usage 2
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

if __name__ == "__main__":
    main()
