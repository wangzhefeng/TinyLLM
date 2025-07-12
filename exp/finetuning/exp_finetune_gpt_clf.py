# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_finetuning_classifier.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-16
# * Version     : 0.1.021612
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
import warnings


import torch

# data
from data_provider.finetune.text_clf.data_loader import create_dataloader
# model
from models.gpt import Model
from model_finetuning.model_finetune_clf import finetune_model
# other model
from layers.tokenizers.tokenization import choose_tokenizer
from model_load.load_gpt2_pretrained_weights import model_with_gpt2_weights, gpt2_model_configs
from model_load.load_pretrained_weights import load_pretrained_model
# training
from utils.llm.calc_loss import calc_loss_batch, calc_loss_loader
from utils.llm.calc_accuracy import calc_accuracy_loader, calc_final_accuracy
from utils.llm.train_funcs import select_optimizer
from utils.plot_losses import plot_values_classifier
# utils
from utils.device import device_setting
from utils.log_util import logger

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelFinetuningClassifier:

    def __init__(self, args):
        self.args = args
        # device
        self.device = device_setting()
        # tokenizer
        self.tokenizer = choose_tokenizer(tokenizer_model = self.args.tokenizer_model)
        # pad token
        self.pad_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special = {"<|endoftext|>"})[0]

    def _get_data(self):
        """
        create dataset and dataloader
        """
        # dataset and dataloader
        self.train_dataset, train_loader = create_dataloader(
            data_path = Path(self.args.data_path).joinpath("train.csv"),
            max_length = None,
            batch_size = self.args.batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = self.args.num_workers,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
        )
        valid_dataset, valid_loader = create_dataloader(
            data_path = Path(self.args.data_path).joinpath("valid.csv"),
            max_length = self.train_dataset.max_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.args.num_workers,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
        )
        test_dataset, test_loader = create_dataloader(
            data_path = Path(self.args.data_path).joinpath("test.csv"),
            max_length = self.train_dataset.max_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.args.num_workers,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
        )
        logger.info(f"train_dataset.max_length: {self.train_dataset.max_length}")
        logger.info(f"valid_dataset.max_length: {valid_dataset.max_length}")
        logger.info(f"test_dataset.max_length: {test_dataset.max_length}")
        logger.info(f"training batches: {len(train_loader)}")
        logger.info(f"validation batches: {len(valid_loader)}")
        logger.info(f"test batches: {len(test_loader)}")

        return train_loader, valid_loader, test_loader

    def _get_model_path(self, setting, training_iter = None):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = Path(self.args.checkpoints).joinpath(setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        best_model_path = f"{model_path}/checkpoint.pth"
        
        return best_model_path
    
    def _get_results_path(self, setting, training_iter):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting).joinpath(str(training_iter))
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def _evaluate_model(self, train_loader, valid_loader, eval_iter: int):
        """
        evaluate model on train and valid_loader
        """
        # eval mode
        self.model.eval()
        
        # calculate loss
        with torch.no_grad():
            train_loss = calc_loss_loader(
                self.args.task_name, 
                train_loader, 
                self.model, 
                self.device, 
                num_batches = eval_iter
            )
            val_loss = calc_loss_loader(
                self.args.task_name, 
                valid_loader, 
                self.model, 
                self.device, 
                num_batches = eval_iter
            )
        
        # train mode
        self.model.train()

        return train_loss, val_loss

    def train(self, training_iter, setting, eval_freq: int = 50, eval_iter: int = 5):
        """
        model training
        """
        # data loader
        torch.manual_seed(self.args.seed)
        train_loader, valid_loader, test_loader = self._get_data()
        
        # load pretained model's weights
        model, base_config = model_with_gpt2_weights(
            cfgs = self.args,
            model_cls = Model, 
            model_source = self.args.pretrained_model_source
        )
        # modify model for finetuning
        torch.manual_seed(self.args.seed)
        self.model = finetune_model(
            model, 
            base_config.emb_dim, 
            self.args.num_classes,
            self.args.finetune_method,
        )
        self.model.to(self.device)
        # optimizer
        self.optimizer = select_optimizer(
            self.model, 
            self.args.learning_rate, 
            self.args.weight_decay
        )

        # checkpoint path
        best_model_path = self._get_model_path(setting)
        # test results path
        results_path = self._get_results_path(setting, training_iter)

        # training start time
        training_start_time = time.time()

        # model training
        # Initialize lists to track losses and examples seen
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
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
                # Calculate loss
                loss = calc_loss_batch(
                    self.args.task_name, 
                    input_batch, 
                    target_batch, 
                    self.model, 
                    self.device
                )
                # Calculate loss gradients
                loss.backward()
                # Update model weights using loss gradients
                self.optimizer.step()
                # track examples instead of tokens
                examples_seen += input_batch.shape[0]
                global_step += 1
                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(train_loader, valid_loader, eval_iter)
                    train_losses.append(train_loss)
                    valid_losses.append(val_loss)
                    logger.info(f"Epoch {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            # Calculate classification accuracy of the model 
            torch.manual_seed(self.args.seed)
            train_accuracy = calc_accuracy_loader(train_loader, self.model, self.device, num_batches=eval_iter)
            val_accuracy = calc_accuracy_loader(valid_loader, self.model, self.device, num_batches=eval_iter)
            logger.info(f"Training accuracy: {train_accuracy*100:.2f}%")
            logger.info(f"Validation accuracy: {val_accuracy*100:.2f}%")
            train_accs.append(train_accuracy)
            valid_accs.append(val_accuracy)
        
        # training end time and training time
        training_end_time = time.time()
        execution_time_minutes = (training_end_time - training_start_time) / 60
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes")
        
        # training loss plot
        plot_values_classifier(
            self.args.train_epochs, 
            examples_seen, 
            train_losses, 
            valid_losses, 
            label = "loss", 
            results_path = results_path,
        )
        # training accuracy plot
        plot_values_classifier(
            self.args.train_epochs, 
            examples_seen, 
            train_accs, 
            valid_accs, 
            label = "accuracy", 
            results_path = results_path,
        )
        # calculate accuracy over complete dataset
        calc_final_accuracy(train_loader, valid_loader, test_loader, self.model, self.device)

    def inference(self, text: str):
        """
        using the LLM as a spam classifier
        """
        # Eval mode
        self.model.eval()

        # Prepare inputs to the model
        input_ids = self.tokenizer.encode(text)
        # Truncate sequences if they too long
        supported_context_length = self.model.pos_emb.weight.shape[0]
        input_ids = input_ids[:min(self.train_dataset.max_length, supported_context_length)]
        # Pad sequences to the longest sequence
        input_ids += [self.pad_token_id] * (self.train_dataset.max_length - len(input_ids))
        # add batch dimension
        input_tensor = torch.tensor(input_ids, device = self.device).unsqueeze(0)

        # Model inference
        with torch.no_grad():
            logits = self.model(input_tensor)[:, -1, :]  # Logits of the last output token

        # probability[optional]
        # probability = torch.softmax(logits, dim = -1)
        # predicted_label = torch.argmax(probability, dim = -1).item()

        # predicted label
        predicted_label = torch.argmax(logits, dim = -1).item()

        # Return the classified result
        classified_result = "spam" if predicted_label == 1 else "not spam"

        return classified_result

    def load_finetuned_model(self, setting, input_text: str):
        # model path
        model_path = self._get_model_path(setting)
        model_path = Path(model_path)
        if not model_path.exists():
            logger.info(f"Could not find '{model_path}'.\n"
                        "Run finetune and save the finetuned model")
        # loade model
        model = load_pretrained_model(
            self.args, 
            model_configs=gpt2_model_configs,
            model_cls = Model, 
            device = self.device, 
            task = "instruction_follow"
        )
        # model inference
        classified_result = self.inference(input_text)

        return classified_result




# 测试代码 main 函数
def main():
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

if __name__ == "__main__":
    main()
