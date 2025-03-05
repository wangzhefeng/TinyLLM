# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_finetuning_instruction.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-16
# * Version     : 0.1.021622
# * Description : supervised instruction finetuning
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import json
import time
from tqdm import tqdm

import torch

# data
from data_provider.finetune.instruction_follow.data_load import load_file
from data_provider.finetune.instruction_follow.data_loader import create_dataloader
from data_provider.finetune.instruction_format import format_input_alpaca
# tokenizer
from tokenizer.tokenization import text_to_token_ids, token_ids_to_text
# model
from models.gpt import Model
from model_train.gpt_generate import generate
# other model
from model_load.openai_gpt2_models import load_pretrained_gpt2_model
# model training
from model_train.calc_loss import _calc_loss_batch, _calc_loss_loader
from model_train.train_funcs import _select_optimizer
from model_train.plot_losses import plot_losses
from model_train.save_load_model import _save_model
# tools
from utils.device import device
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ModelFinetuningInstructionFlow:

    def __init__(self, args):
        super(ModelFinetuningInstructionFlow, self).__init__()
        self.args = args

    def _build_data(self):
        """
        create dataset and dataloader
        """
        # data
        data = load_file(file_path = self.args.data_path)
        logger.info(f"Number of entries: {len(data)}")
        # data split ratio
        train_portion = int(len(data) * self.args.train_ratio)
        test_portion = int(len(data) * self.args.test_ratio)
        valid_portion = len(data) - train_portion - test_portion
        # data split
        train_data = data[:train_portion]
        test_data = data[train_portion:(train_portion + test_portion)]
        valid_data = data[(train_portion + test_portion):]
        # logger.info(f"train_data: \n{train_data[0]}")
        # logger.info(f"test_data: \n{test_data[0]}")
        # logger.info(f"valid_data: \n{valid_data[0]}")
        logger.info(f"Training data length: {len(train_data)}")
        logger.info(f"Test data length: {len(test_data)}")
        logger.info(f"Validation data length: {len(valid_data)}")
        # dataset and dataloader
        train_dataset, train_dataloader = create_dataloader(
            data = train_data,
            batch_size = self.args.batch_size,
            shuffle = True,
            drop_last = True,
        )
        test_dataset, test_dataloader = create_dataloader(
            data = test_data,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
        )
        valid_dataset, valid_dataloader = create_dataloader(
            data = valid_data,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
        )

        return (
            train_data, train_dataset, train_dataloader, 
            test_data, test_dataset, test_dataloader,
            valid_data, valid_dataset, valid_dataloader,
        )

    # ------------------------------
    # Finetuning LLM on instruction data
    # ------------------------------
    def valid(self, train_loader, val_loader, eval_iter):
        """
        model evaluate
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

    def train(self, valid_data, eval_freq: int = 5, eval_iter: int = 5):
        """
        model training
        """
        # data loader
        train_loader, valid_loader, test_loader = self._build_data()
        # model
        self.model, self.base_config = load_pretrained_gpt2_model(cfgs=self.args, model_cls=Model)
        # move model to device
        self.model.to(self.device)
        # optimizer
        self.optimizer = _select_optimizer(self.model)

        # training start time
        training_start_time = time.time()
         
        # model training
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses = [], []
        track_tokens_seen = []
        tokens_seen = 0
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
                # track examples instead of tokens
                tokens_seen += input_batch.numel()
                global_step += 1
                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.valid(train_loader, valid_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    logger.info(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            start_context = format_input_alpaca(valid_data[0])
            generate(self.model, self.device, start_context)
        
        # training end time
        training_end_time = time.time()
        # training time
        execution_time_minutes = (training_end_time - training_start_time) / 60
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

        # plot losses
        epochs_tensor = torch.linspace(0, self.args.train_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    def _extract_save_responses(self, test_data):
        """
        Extracting and saving responses
        """
        torch.manual_seed(123)
        for entry in test_data[:3]:
            input_text = format_input_alpaca(entry)
            token_ids = generate(
                model = self.model,
                token_idx = text_to_token_ids(input_text).to(self.device),
                max_new_tokens = 256,
                context_size = self.base_config.context_length,
                eos_id = 50256
            )
            generated_text = token_ids_to_text(token_ids)
            response_text = (
                generated_text[len(input_text):]
                .replace("### Response:", "")
                .strip()
            )
            logger.info(input_text)
            logger.info(f"\nCorrect response:\n>> {entry['output']}")
            logger.info(f"\nModel response:\n>> {response_text.strip()}")
            logger.info("-------------------------------------")

        for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
            input_text = format_input_alpaca(entry)
            token_ids = generate(
                model = self.model,
                token_idx = text_to_token_ids(input_text).to(self.device),
                max_new_tokens = 256,
                context_size = self.base_config.context_length,
                eos_id = 50256
            )
            generated_text = token_ids_to_text(token_ids)
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
            test_data[i]["model_response"] = response_text

    def _build_test_data(self, test_data):
        """
        build instruction data with response
        """
        result_path = "./saved_results/test_results/"
        os.makedirs(result_path, exist_ok=True)
        with open(os.path.join(result_path, "instruction-data-with-response.json"), "w") as file:
            json.dump(test_data, file, indent=4)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
