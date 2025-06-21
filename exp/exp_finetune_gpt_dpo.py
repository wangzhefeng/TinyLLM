# -*- coding: utf-8 -*-

# ***************************************************
# * File        : preference_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2025-02-22
# * Version     : 0.1.022201
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
import re
import time
from typing import Dict


import torch
import torch.nn.functional as F
from transformers import GPT2Model

# data
from data_provider.finetune.dpo import data_loader
from data_provider.finetune import instruction_format
# model
from models.gpt import Model
from utils.train_utils.gpt_generate import generate
# tokenzier
from tokenizer.tokenization import choose_tokenizer, text_to_token_ids, token_ids_to_text
from model_load.load_gpt2_pretrained_weights import (
    gpt2_model_configs, 
    gpt2_huggingface_models,
)
from model_load.openai_gpt2_weights_load_hf import load_weights_hf
from utils.train_utils.train_funcs import select_optimizer
from utils.train_utils.plot_losses import plot_losses
# utils
from utils.device import device_setting
from utils.args_tools import DotDict
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelFinetuningPreference:

    def __init__(self, args):
        self.args = args
        # device
        self.device = device_setting()
        # tokenizer
        self.tokenizer = choose_tokenizer(tokenizer_model = self.args.tokenizer_model)
        # pad token id
        self.pad_token_id = self.tokenizer.encode("<|endoftext|>", allowed_special = {"<|endoftext|>"})[0]

    def _get_data(self):
        # data load and split
        train_data, test_data, valid_data = data_loader.load_split_data(
            self.args.data_path,
            self.args.train_ratio,
            self.args.test_ratio,
        )
        # dataset and dataloader
        train_dataset, train_dataloader = data_loader.create_dataloader(
            data = train_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            mask_prompt_tokens = self.args.mask_prompt_tokens,
            allowed_max_length = self.args.context_length,
            batch_size = self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers = self.args.num_workers,
        )
        test_dataset, test_dataloader = data_loader.create_dataloader(
            data = test_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            mask_prompt_tokens = self.args.mask_prompt_tokens,
            allowed_max_length = self.args.context_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.args.num_workers, 
        )
        valid_dataset, valid_dataloader = data_loader.create_dataloader(
            data = valid_data,
            device = self.device,
            tokenizer = self.tokenizer,
            pad_token_id = self.pad_token_id,
            mask_prompt_tokens = self.args.mask_prompt_tokens,
            allowed_max_length = self.args.context_length,
            batch_size = self.args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self.args.num_workers, 
        )
        logger.info(f"Training data length: {len(train_data)} Training batches: {len(train_dataloader)}")
        logger.info(f"Test data length: {len(test_data)} Test batches: {len(test_dataloader)}")
        logger.info(f"Valid data length: {len(valid_data)} Validation batches: {len(valid_dataloader)}")

        return (
            train_data, train_dataset, train_dataloader, 
            test_data, test_dataset, test_dataloader,
            valid_data, valid_dataset, valid_dataloader,
        )

    def __load_finetuned_model(self, model_path, update_params: bool = True, base_config_new: Dict = None):
        """
        initializing a model with pretrained weights
        """
        # model base config 
        base_config = {
            "vocab_size": self.args.vocab_size,          # Vocabulary size: 50257
            "context_length": self.args.context_length,  # Context length: 1024
            "dropout": self.args.dropout,                # Dropout rate: 0.0
            "qkv_bias": self.args.qkv_bias,              # Query-key-value bias: True 
        }
        # update model base config
        if update_params:
            # huggingface gpt2 pretrained model
            gpt2_hf = GPT2Model.from_pretrained(
                gpt2_huggingface_models[self.args.pretrained_model],
                cache_dir = self.args.pretrained_model_path,
            )
            gpt2_hf.eval()
            # update pretrained model's config 
            base_config.update(gpt2_model_configs[self.args.pretrained_model])
            base_config = DotDict(base_config)
            # pretrained model instance
            model = Model(base_config)
        else:
            model = Model(base_config_new)
        # load model weights
        model.load_state_dict(torch.load(
            model_path,
            map_location = torch.device("cpu"), 
            weights_only = True
        ))
        # assign pretrained model's weights
        if update_params:
            load_weights_hf(model, gpt2_hf, base_config)
        # model inference mode
        model.eval()

        return model, base_config

    def __load_dpo_finetuned_model(self, model_path):
        """
        initializing a model with pretrained weights
        """
        # model base config
        base_config = {
            "vocab_size": self.args.vocab_size,          # Vocabulary size: 50257
            "context_length": self.args.context_length,  # Context length: 1024
            "dropout": self.args.dropout,                # Dropout rate: 0.0
            "qkv_bias": self.args.qkv_bias,              # Query-key-value bias: True
            "emb_dim": self.args.emb_dim,                # Embedding dimension: 768
            "n_layers": self.args.n_layers,              # Number of layers: 12
            "n_heads": self.args.n_heads,                # Number of heads: 12
        }
        base_config = DotDict(base_config)
        model = Model(base_config)
        # load model weights
        model.load_state_dict(torch.load(
            model_path,
            map_location = torch.device("cpu"), 
            weights_only = True
        ))
        # model inference mode
        model.eval()

        return model

    def _build_policy_reference_model(self, model_path):
        # 策略模型(希望优化的模型)
        policy_model, base_config = self.__load_finetuned_model(
            model_path,
            update_params=True
        )
        policy_model.to(self.device)
        # 参考模型(保持不变的原始模型)
        reference_model, base_config = self.__load_finetuned_model(
            model_path,
            update_params=False, 
            base_config_new=base_config
        )
        reference_model.to(self.device)

        return policy_model, reference_model

    def _get_model_path(self, setting, training_iter: int = None):
        """
        模型保存路径
        """
        # 模型保存路径
        model_dir = Path(self.args.model_path).joinpath(setting).joinpath(str(training_iter))
        os.makedirs(model_dir, exist_ok=True)
        # 最优模型保存路径
        self.model_path = Path(model_dir).joinpath(f"{re.sub(r'[ ()]', '', self.args.pretrained_model) }-sft.pth")
        # best_model_path = f"{self.model_dir}/checkpoint.pth"
        # return best_model_path
    
    def _get_results_path(self, setting, training_iter):
        """
        结果保存路径
        """
        results_path = Path(self.args.test_results).joinpath(setting).joinpath(str(training_iter))
        os.makedirs(results_path, exist_ok=True)
        
        return results_path
    
    def _save_model(self, model):
        """
        Save model
        """
        torch.save(model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def _load_model(self, model):
        """
        Load model
        """
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        logger.info(f"Model loaded from {self.model_path}")
        
        return model
 
    def __compute_dpo_loss(self, 
                          model_chosen_logprobs,
                          model_rejected_logprobs,
                          reference_chosen_logprobs,
                          reference_rejected_logprobs,
                          beta = 0.1):
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. 
                Shape: (batch_size,)
            policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. 
                Shape: (batch_size,)
            reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. 
                Shape: (batch_size,)
            reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses.
                Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. 
            We ignore the reference model as beta -> 0.
            label_smoothing: conservativeness for DPO loss.

        Returns:
            A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
        """
        model_logratios = model_chosen_logprobs - model_rejected_logprobs
        reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
        logits = model_logratios - reference_logratios
        # DPO（参见 https://arxiv.org/pdf/2305.18290.pdf 中的公式 7）
        losses = -F.logsigmoid(beta * logits)
        # 可选值，用于在训练期间跟踪进度
        chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
        rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()
        # 使用 .mean() 对批次中的样本进行平均
        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def __compute_logprobs(self, logits, labels, selection_mask=None):
        """
        Compute log probabilities.

        Args:
            logits: Tensor of shape (batch_size, num_tokens, vocab_size)
            labels: Tensor of shape (batch_size, num_tokens)
            selection_mask: Tensor for shape (batch_size, num_tokens)

        Returns:
            mean_log_prob: Mean log probability excluding padding tokens.
        """
        # 标签是输入向右移动一位
        labels = labels[:, 1:].clone()
        # 截断 Logits 以匹配标签的token数量
        logits = logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        # 收集实际标签的对数概率
        selected_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        if selection_mask is not None:
            mask = selection_mask[:, 1:].clone()
            # 应用掩码以过滤掉填充token
            selected_log_probs = selected_log_probs * mask
            # 计算排除填充token的平均对数概率
            # 这是在token上取平均，因此形状为 (batch_size, num_tokens)
            avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
            return avg_log_prob
        else:
            return selected_log_probs.mean(-1)

    def __compute_dpo_loss_batch(self, batch, policy_model, reference_model, beta):
        """
        Compute the DPO loss on an input batch
        """
        # 其中 policy_model(batch["chosen"]) 是 logits
        policy_chosen_log_probas = self.__compute_logprobs(
            logits=policy_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        policy_rejected_log_probas = self.__compute_logprobs(
            logits=policy_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )
        with torch.no_grad():
            ref_chosen_log_probas = self.__compute_logprobs(
                logits=reference_model(batch["chosen"]),
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )
            ref_rejected_log_probas = self.__compute_logprobs(
                logits=reference_model(batch["rejected"]),
                labels=batch["rejected"],
                selection_mask=batch["rejected_mask"]
            )
        loss, chosen_rewards, rejected_rewards = self.__compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas,
            beta=beta
        )

        return loss, chosen_rewards, rejected_rewards

    def __compute_dpo_loss_loader(self, data_loader, policy_model, reference_model, beta, num_batches=None):
        """
        Apply compute_dpo_loss_batch to a whole data loader
        """
        total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # 如果指定的批次数量超过了数据加载器中的批次数量，则减少批次数量以匹配数据加载器中的总批次数量
            num_batches = min(num_batches, len(data_loader))
        for i, batch in enumerate(data_loader):
            if i < num_batches:
                loss, chosen_rewards, rejected_rewards = self.__compute_dpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    beta=beta
                )
                total_loss += loss.item()
                total_chosen_rewards += chosen_rewards.item()
                total_rejected_rewards += rejected_rewards.item()
            else:
                break
        # 计算平均值
        total_loss /= num_batches
        total_chosen_rewards /= num_batches
        total_rejected_rewards /= num_batches

        return total_loss, total_chosen_rewards, total_rejected_rewards

    def __evaluate_dpo_loss_loader(self, policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
        """
        Compute the DPO loss for the training and validation dataset
        """
        policy_model.eval()
        with torch.no_grad():
            train_loss, train_chosen_rewards, train_rejected_rewards = self.__compute_dpo_loss_loader(
                data_loader=train_loader,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
                num_batches=eval_iter
            )
            valid_loss, valid_chosen_rewards, valid_rejected_rewards = self.__compute_dpo_loss_loader(
                data_loader=val_loader,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
                num_batches=eval_iter
            )
        res = {
            "train_loss": train_loss,
            "train_chosen_reward": train_chosen_rewards,
            "train_rejected_reward": train_rejected_rewards,
            "valid_loss": valid_loss,
            "valid_chosen_reward": valid_chosen_rewards,
            "valid_rejected_reward": valid_rejected_rewards
        }
        policy_model.train()

        return res

    def train(self, training_iter, setting, eval_freq: int = 5, eval_iter: int = 5):
        # load data
        (
            train_data, train_dataset, train_loader, 
            test_data, test_dataset, test_loader,
            valid_data, valid_dataset, valid_loader,
        ) = self._get_data()
        
        # load model
        self.policy_model, self.reference_model = self._build_policy_reference_model(
            model_path=self.args.finetuned_model_path
        )
        
        # optimizer
        self.optimizer = select_optimizer(
            self.policy_model,
            self.args.learning_rate,
            self.args.weight_decay
        )

        # checkpoint path
        self._get_model_path(setting, training_iter)

        # test results path
        results_path = self._get_results_path(setting, training_iter)

        # record training start time
        training_start_time = time.time()

        # 初始化列表以跟踪损失和已处理的token
        tracking = {
            "train_losses": [],
            "train_chosen_rewards": [],
            "train_rejected_rewards": [],
            "valid_losses": [],
            "valid_chosen_rewards": [],
            "valid_rejected_rewards": [],
            "tokens_seen": []
        }
        tokens_seen = 0
        global_step = -1
        # 主训练循环
        for epoch in range(self.args.train_epochs):
            # 将模型设置为训练模式
            self.policy_model.train()
            for batch_idx, batch in enumerate(train_loader):
                # 重置上一批次的损失梯度
                self.optimizer.zero_grad()
                # 计算 DPO 损失
                loss, \
                chosen_rewards, \
                rejected_rewards = self.__compute_dpo_loss_batch(
                    batch = batch,
                    policy_model = self.policy_model,
                    reference_model = self.reference_model,
                    beta=self.args.beta  # 取值在0.1到0.5之间
                )
                # 计算损失梯度
                loss.backward()
                # 使用损失梯度更新模型权重
                self.optimizer.step()
                # 记录损失和已处理的 token
                tokens_seen += batch["chosen"].numel()
                global_step += 1
                # 可选的评估步骤
                if global_step % eval_freq == 0:
                    res = self.__evaluate_dpo_loss_loader(
                        policy_model = self.policy_model,
                        reference_model = self.reference_model,
                        train_loader = train_loader,
                        val_loader = valid_loader,
                        beta = self.args.beta,
                        eval_iter = eval_iter
                    )
                    tracking["train_losses"].append(res["train_loss"])
                    tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                    tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                    tracking["valid_losses"].append(res["valid_loss"])
                    tracking["valid_chosen_rewards"].append(res["valid_chosen_reward"])
                    tracking["valid_rejected_rewards"].append(res["valid_rejected_reward"])
                    tracking["tokens_seen"].append(tokens_seen)
                    train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                    val_reward_margin = res["valid_chosen_reward"] - res["valid_rejected_reward"]
                    logger.info(
                        f"Epoch {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {res['train_loss']:.3f}, Valid loss {res['valid_loss']:.3f}, "
                        f"Train reward margins {train_reward_margin:.3f}, "
                        f"Valid reward margins {val_reward_margin:.3f}"
                    ) 
        # training end time
        training_end_time = time.time()
        execution_time_minutes = (training_end_time - training_start_time) / 60
        logger.info(f"Training completed in {execution_time_minutes:.2f} minutes")
        # plot DPO losses
        plot_losses(
            self.args.train_epochs, 
            tokens_seen = tracking["tokens_seen"], 
            train_losses = tracking["train_losses"], 
            val_losses = tracking["valid_losses"],
            label = "loss",
            results_path = results_path
        )
        # plot reward margins
        train_reward_margins = [i-j for i, j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
        valid_reward_margins = [i-j for i, j in zip(tracking["valid_chosen_rewards"], tracking["valid_rejected_rewards"])]
        plot_losses(
            self.args.train_epochs, 
            tokens_seen = tracking["tokens_seen"], 
            train_losses = train_reward_margins, 
            val_losses = valid_reward_margins, 
            label = "reward margins",
            results_path = results_path,
        )
        # model save
        self._save_model(self.policy_model)
     
    def __extract_response(self, model, entry):
        """
        提取模型响应
        """
        # 在每个训练周期后打印示例文本
        input_text = instruction_format.format_input_alpaca(entry)
        logger.info(f"Input text: {input_text}")
        token_ids = generate(
            model = model,
            token_idx = text_to_token_ids(input_text).to(self.device),
            max_new_tokens = self.args.max_new_tokens,  # 256
            context_size = self.args.context_length,
            temperature=self.args.temperature,
            top_k = self.args.top_k,
            eos_id = self.pad_token_id,
        )
        generated_text = token_ids_to_text(token_ids)
        # 对响应进行清理，只返回响应文本，并去除提示和提示样式
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        
        return response_text

    def inference(self, training_iter, setting, data_name: str = "valid", data_index: int = 3):
        """
        model inference
        """
        # load data
        (
            train_data, train_dataset, train_loader, 
            test_data, test_dataset, test_loader,
            valid_data, valid_dataset, valid_loader,
        ) = self._get_data()
        # load model
        self._get_model_path(setting, training_iter)
        policy_model, reference_model = self._build_policy_reference_model(model_path=self.model_path)
        # choose data
        if data_name == "valid":
            data = valid_data
        elif data_name == "test":
            data = test_data
        # print response
        for entry in data[:data_index]:
            reference_response_text = self.__extract_response(policy_model, entry)
            policy_response_text = self.__extract_response(reference_model, entry)
            logger.info(f"\nCorrect response:\n>> {entry['output']}")
            logger.info(f"\nReference response:\n>> {reference_response_text.strip()}")
            logger.info(f"\nPolicy response:\n>> {policy_response_text.strip()}")
            logger.info(f"\n{40 * '-'}")




# 测试代码 main 函数
def main():
    # model test before dpo
    prompt = """Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    Convert the active sentence to passive: 'The chef cooks the meal every day.'
    """

if __name__ == "__main__":
    main()
