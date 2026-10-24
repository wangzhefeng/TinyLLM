# -*- coding: utf-8 -*-

# ***************************************************
# * File        : gpt_finetuning_instruction.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-16
# * Version     : 0.1.021622
# * Description : supervised instruction finetuning
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import json
import time
import warnings

from tqdm import tqdm

import torch

# data
from post_training.sft.data_provider.instruction_follow import data_loader
from layers.tokenizers import chat as instruction_format
# tokenizer
from layers.tokenizers.tokenization import choose_tokenizer
from layers.tokenizers.tokenization import text_to_token_ids, token_ids_to_text
# model
from models.gpt2.gpt2_124M import Model
# other model
from models.io.gpt2.load_gpt2_pretrained_weights import (
    model_with_gpt2_weights, 
    gpt2_model_configs
)
from models.io.gpt2.load_pretrained_weights import load_pretrained_model
# model training
from utils.calc_loss import calc_loss_batch, calc_loss_loader
from utils.optimizer import select_optimizer
from training.trainer import Trainer
from training.checkpoint import load_training_checkpoint, save_training_checkpoint
from training.experiment import experiment_paths, results_paths
from utils.plot_losses import plot_losses_llm
# tools
from utils.device import device_setting
from utils.log_util import logger

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelFinetuningInstructionFlow:

    def __init__(self, args):
        self.args = args
        # device
        self.device = device_setting()
        # tokenizer
        self.tokenizer = choose_tokenizer(tokenizer_model = self.args.tokenizer_model)
        # pad token
        self.pad_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special = {"<|endoftext|>"})[0]
        self.generator = None

    def _get_data(self):
        """
        create dataset and dataloader
        """
        # data load and split
        train_data, test_data, valid_data = data_loader.load_split_data(
            self.args.data_path, 
            self.args.train_ratio, 
            self.args.test_ratio,
        )
        # dataset and dataloader
        torch.manual_seed(self.args.seed)
        train_dataset, train_dataloader = data_loader.create_dataloader(
            data = train_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            ignore_index = -100,
            allowed_max_length = 1024,
            batch_size = self.args.batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = self.args.num_workers
        )
        test_dataset, test_dataloader = data_loader.create_dataloader(
            data = test_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            ignore_index = -100,
            allowed_max_length = 1024,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.args.num_workers
        )
        valid_dataset, valid_dataloader = data_loader.create_dataloader(
            data = valid_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            ignore_index = -100,
            allowed_max_length = 1024,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.args.num_workers
        )
        logger.info(f"Training data length: {len(train_data)} Training batches: {len(train_dataloader)}")
        logger.info(f"Test data length: {len(test_data)} Test batches: {len(test_dataloader)}")
        logger.info(f"Valid data length: {len(valid_data)} Validation batches: {len(valid_dataloader)}")

        return (
            train_data, train_dataset, train_dataloader, 
            test_data, test_dataset, test_dataloader,
            valid_data, valid_dataset, valid_dataloader,
        )
    
    def _get_model_path(self, setting, training_iter: int = None):
        """
        模型保存路径
        """
        model_dir = experiment_paths(self.args.finetuned_model_path, setting, training_iter)
        self.model_path = model_dir / f"{re.sub(r'[ ()]', '', self.args.pretrained_model)}-sft.pth"

    def _load_backbone(self):
        """
        加载预训练底座：默认 GPT-2 官方权重；
        ``args.backbone == "phi15"`` 时改用 phi-1.5（LLMs-from-scratch ch07 路线）。
        """
        if getattr(self.args, "backbone", "gpt2") == "phi15":
            from models.io.phi.load_phi15_weights import model_with_phi15_weights

            model, base_config = model_with_phi15_weights(
                model_path = getattr(self.args, "pretrained_model_path", None),
            )
        else:
            model, base_config = model_with_gpt2_weights(
                cfgs = self.args,
                model_cls = Model,
                model_source = self.args.pretrained_model_source,
            )
        model = self._load_model(model)
        model.to(self.device)

        return model, base_config

    def _get_results_path(self, setting, training_iter):
        return results_paths(self.args.test_results, setting, training_iter)

    def _save_model(self, model):
        save_training_checkpoint(self.model_path, model=model, optimizer=getattr(self, "optimizer", None))
        logger.info(f"Model saved to {self.model_path}")

    def _load_model(self, model):
        load_training_checkpoint(self.model_path, model=model, map_location=self.device)
        logger.info(f"Model loaded from {self.model_path}")
        
        return model

    def valid(self, train_loader, val_loader, eval_iter: int):
        """
        model evaluate
        """
        # eval mode
        self.model.eval()
        # calculate loss
        with torch.no_grad():
            train_loss = calc_loss_loader(
                self.args.task_name, 
                train_loader, 
                self.model,
                device=self.device,
                num_batches=eval_iter
            )
            val_loss = calc_loss_loader(
                self.args.task_name, 
                val_loader, 
                self.model,
                device=self.device,
                num_batches=eval_iter
            )
        # train mode
        self.model.train()

        return train_loss, val_loss

    def train(self, training_iter, setting, eval_freq: int = 5, eval_iter: int = 5):
        """
        model training
        """
        # data loader
        (
            train_data, train_dataset, train_loader, 
            test_data, test_dataset, test_loader,
            valid_data, valid_dataset, valid_loader,
        ) = self._get_data()

        # model
        self.model, self.base_config = model_with_gpt2_weights(
            cfgs = self.args, 
            model_cls = Model, 
            model_source = self.args.pretrained_model_source
        )
        # move model to device
        self.model.to(self.device)
        
        # optimizer
        self.optimizer = select_optimizer(
            self.model,
            self.args.learning_rate,
            self.args.weight_decay
        )
        trainer = Trainer(self.model, self.optimizer, self.device, grad_clip=0.0)

        def train_loss_fn(batch):
            input_ids, targets = batch
            return calc_loss_batch(
                self.args.task_name, input_ids, targets, self.model, device=self.device
            )

        # checkpoint path
        self._get_model_path(setting, training_iter)
        # test results path
        results_path = self._get_results_path(setting, training_iter)

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
            for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                trainer.train_step(
                    (input_batch, target_batch),
                    train_loss_fn,
                    batch_idx=batch_idx,
                    num_batches=len(train_loader),
                )
                # track examples instead of tokens
                tokens_seen += input_batch.numel()
                global_step += 1
                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.valid(train_loader, valid_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    logger.info(f"Epoch {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # 可选采样由入口注入；未配置时不影响训练
            if self.generator is not None:
                self.model.eval()
                start_context = valid_data[0]
                formated_start_context = instruction_format.format_input_alpaca(start_context)
                token_ids = self.generator(
                    model = self.model,
                    token_ids = text_to_token_ids(formated_start_context).to(self.device),
                    max_new_tokens = self.args.max_new_tokens,
                    context_length = self._context_length(),
                    eos_token_id = self.pad_token_id
                )
                generated_text = token_ids_to_text(token_ids)
                response_text = (
                    generated_text[len(formated_start_context):]
                    .replace("### Response:", "")
                    .strip()
                )
                logger.info(f"Epoch {epoch+1}: start_context: \n{formated_start_context}")
                logger.info(f"Epoch {epoch+1}: response_text: \n{response_text}")
                self.model.train()
        
        # training end time
        training_end_time = time.time()
        # training time
        execution_time_minutes = (training_end_time - training_start_time) / 60
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes")
        
        # plot losses（tokens_seen 版本对应 plot_losses_llm，见 utils/plot_losses.py）
        plot_losses_llm(
            train_epochs=self.args.train_epochs, 
            tokens_seen=track_tokens_seen, 
            train_losses=train_losses, 
            valid_losses=val_losses, 
            label="loss",
            results_path=results_path
        )
        
        # model save
        self._save_model(self.model)
        
        # parpare eval data
        if self.args.is_eval:
            self.eval_data_papare(test_data)

    def test(self, test_entry_num: int = 3, training_iter = None, setting = None):
        """
        Extracting and saving responses
        """ 
        # load data
        (
            train_data, train_dataset, train_loader, 
            test_data, test_dataset, test_loader,
            valid_data, valid_dataset, valid_loader,
        ) = self._get_data()
        
        self._get_model_path(setting, training_iter)
        model, base_config = self._load_backbone()
        
        # inference
        torch.manual_seed(self.args.seed)
        for entry in test_data[:test_entry_num]:
            input_text = instruction_format.format_input_alpaca(entry)
            token_ids = self.generator(
                model = model,
                token_ids = text_to_token_ids(input_text).to(self.device),
                max_new_tokens = self.args.max_new_tokens,
                context_length = self._context_length(),
                eos_token_id = self.pad_token_id
            )
            generated_text = token_ids_to_text(token_ids)
            response_text = (
                generated_text[len(input_text):]
                .replace("### Response:", "")
                .strip()
            )
            logger.info("-------------------------------------")
            logger.info(f"input_text:\n{input_text}")
            logger.info(f"\nCorrect response:\n>> {entry['output']}")
            logger.info(f"\nModel response:\n>> {response_text.strip()}")

    def eval_data_papare(self, eval_data):
        """
        build instruction data with response
        """
        model, base_config = self._load_backbone()

        # data process
        for i, entry in tqdm(enumerate(eval_data), total = len(eval_data)):
            input_text = instruction_format.format_input_alpaca(entry)
            token_ids = self.generator(
                model = model,
                token_ids = text_to_token_ids(input_text).to(self.device),
                max_new_tokens = self.args.max_new_tokens,
                context_length = self._context_length(),
                eos_token_id = self.pad_token_id
            )
            generated_text = token_ids_to_text(token_ids)
            response_text = (
                generated_text[len(input_text):]
                .replace("### Response:", "")
                .strip()
            )
            eval_data[i]["model_response"] = response_text
        # save test data
        with open(self.args.eval_data_path, "w") as file:
            json.dump(eval_data, file, indent = 4)

    def inference(self, entry, training_iter = None,setting = None):
        self._get_model_path(setting, training_iter)
        model, base_config = self._load_backbone()

        # inference
        input_text = instruction_format.format_input_alpaca(entry)
        token_ids = self.generator(
            model = model,
            token_ids = text_to_token_ids(input_text).to(self.device),
            max_new_tokens = self.args.max_new_tokens,
            context_length = self._context_length(),
            eos_token_id = self.pad_token_id
        )
        generated_text = token_ids_to_text(token_ids)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        logger.info("-------------------------------------")
        logger.info(f"input_text:\n{input_text}")
        logger.info(f"\nCorrect response:\n>> {entry['output']}")
        logger.info(f"\nModel response:\n>> {response_text.strip()}")

    def _context_length(self):
        """
        上下文长度：优先用模型实际 base_config（train 加载后存在），
        未训练直接 test/inference 时回退到 args.context_length。
        """
        if getattr(self, "base_config", None) is not None:
            return self.base_config.context_length
        return self.args.context_length

    def _extract_response(self, response_text, input_text):
        """
        提取 response_text 中的 response
        """
        response = response_text[len(input_text):] \
            .replace("### Response:", "") \
            .strip()
        
        return response

    def load_finetuned_model(self, setting, train_iter, prompt):
        """
        加载微调好的模型，并使用模型进行推断

        Args:
            setting (_type_): _description_
            train_iter (_type_): _description_
            prompt (_type_): _description_

        Returns:
            _type_: _description_
        """
        # model path
        model_path = self._get_model_path(setting, train_iter)
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.info(f"Could not find '{model_path}'.\n"
                         "Run finetune and save the finetuned model")
        # load model
        model = load_pretrained_model(
            self.args, 
            model_configs=gpt2_model_configs,
            model_cls = Model, 
            device = torch.device("cpu"), 
            task = "instruction_follow"
        )
        # inference
        token_ids = self.generator(
            model = model,
            token_ids = text_to_token_ids(prompt),
            max_new_tokens = self.args.max_new_tokens,  # 35
            context_length = self._context_length(),
            eos_token_id = self.pad_token_id
        )
        response = self._extract_response(
            response_text = token_ids_to_text(token_ids), 
            input_text = prompt,
        )
        
        return response




# 测试代码 main 函数
def main():
    prompt = """Below is an instruction that describes a task. Write a response 
    that appropriately completes the request.

    ### Instruction:
    Convert the active sentence to passive: 'The chef cooks the meal every day.'
    """

if __name__ == "__main__":
    main()
