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
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import time

import torch
import torch.nn.functional as F

from finetuning.direct_preference_optimization.data_load import load_instruction_data
from finetuning.instruction_format import format_input_alpaca
from finetuning.direct_preference_optimization.data_loader import create_dataloader
from models.gpt import Model
from training.generate import _generate_and_print_sample
from utils.argsparser_tools import DotDict
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def _build_data(data_path: str, train_ratio: float = 0.85, test_ratio: float = 0.10, batch_size: int = 2):
    # data load
    data = load_instruction_data(data_path = data_path)
    logger.info(f"Number of entries: {len(data_path)}")
    logger.info(f"data[50]: {data[50]}")
    logger.info(f"data[999]: {data[999]}")
    # data format
    model_input = format_input_alpaca(data[50])
    logger.info(f"model_input: {model_input}")
    # data view
    desired_response = f"### Response:\n{data[50]['chosen']}"
    possible_response = f"### Response:\n{data[50]['rejected']}"
    logger.info(f"desired_response: {desired_response}")
    logger.info(f"possible_response: {possible_response}")
    # data split
    train_portion = int(len(data) * train_ratio)  # 85% 用作训练集
    test_portion = int(len(data) * test_ratio)    # 10% 用作测试集
    val_portion = len(data) - train_portion - test_portion  # 剩下的 5% 用作验证集
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    valid_data = data[train_portion + test_portion:]
    logger.info(f"Train set length: {len(train_data)}")
    logger.info(f"Test set length: {len(test_data)}")
    logger.info(f"Validation set length: {len(valid_data)}")
    # dataset and dataloader
    train_dataset, train_dataloader = create_dataloader(
        train_data, 
        batch_size = batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_dataset, test_dataloader = create_dataloader(
        data = test_data,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
    )
    valid_dataset, valid_dataloader = create_dataloader(
        data = valid_data,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
    )

    return (
        train_data, train_dataset, train_dataloader, 
        test_data, test_dataset, test_dataloader,
        valid_data, valid_dataset, valid_dataloader,
    )


def _build_model(model_path):
    # Loading pretrained LLM   
    choose_model = "gpt2-medium (355M)"
    # model base cofig
    base_config = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "dropout": 0.0,          # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    # huggingface allowed model names  
    gpt2_model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    base_config.update(gpt2_model_configs[choose_model])
    base_config = DotDict(base_config)
    # custom gpt model
    model = Model(base_config)
    # model weight load
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval();

    return model, base_config


def _build_policy_reference_model(model, device, base_config, finetuned_model_path):
    # 策略模型(希望优化的模型)
    policy_model = model
    policy_model.to(device)

    # 参考模型(保持不变的原始模型)
    reference_model = Model(base_config)
    reference_model.load_state_dict(torch.load(
        finetuned_model_path, 
        map_location = torch.device("cpu"), 
        weights_only = True
    ))
    reference_model.eval()
    reference_model.to(device)

    return policy_model, reference_model


# TODO
# def extract_response(response_text, input_text):
#     """
#     对响应进行清理，只返回响应文本，并去除提示和提示样式    
#     """
#     response = response_text[len(input_text):].replace("### Response:", "").strip()

#     return response


def _compute_dpo_loss(model_chosen_logprobs,
                     model_rejected_logprobs,
                     reference_chosen_logprobs,
                     reference_rejected_logprobs,
                     beta=0.1):
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


def _compute_logprobs(logits, labels, selection_mask=None):
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


def _compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """
    Compute the DPO loss on an input batch
    """
    # 其中 policy_model(batch["chosen"]) 是 logits
    policy_chosen_log_probas = _compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    policy_rejected_log_probas = _compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )
    with torch.no_grad():
        ref_chosen_log_probas = _compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        ref_rejected_log_probas = _compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )
    loss, chosen_rewards, rejected_rewards = _compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )

    return loss, chosen_rewards, rejected_rewards


def _compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
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
            loss, chosen_rewards, rejected_rewards = _compute_dpo_loss_batch(
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


def _evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """
    Compute the DPO loss for the training and validation dataset
    """
    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = _compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )
        val_loss, val_chosen_rewards, val_rejected_rewards = _compute_dpo_loss_loader(
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
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }
    policy_model.train()

    return res


def _train_model_dpo_simple(model, 
                           policy_model, 
                           reference_model, 
                           train_loader, 
                           val_loader,
                           optimizer, 
                           num_epochs, 
                           beta,
                           eval_freq, 
                           eval_iter, 
                           start_context, 
                           tokenizer):
    # 初始化列表以跟踪损失和已处理的token
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1
    # 主训练循环
    for epoch in range(num_epochs):
        policy_model.train()  # 将模型设置为训练模式
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()  # 重置上一批次的损失梯度
            loss, chosen_rewards, rejected_rewards = _compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 使用损失梯度更新模型权重
            tokens_seen += batch["chosen"].numel()
            global_step += 1
            # 可选的评估步骤
            if global_step % eval_freq == 0:
                res = _evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]
                logger.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )
        # 在每个训练周期后打印示例文本
        _generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=loss.device,
            start_context=start_context
        )

    return tracking


def train(policy_model, reference_model, train_loader, val_loader, val_data, tokenizer):
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)
    num_epochs = 1
    tracking = _train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=0.1, # 取值在0.1到0.5之间
        eval_freq=5,
        eval_iter=5,
        start_context=format_input_alpaca(val_data[2]),
        tokenizer=tokenizer
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")




# 测试代码 main 函数
def main():
    # params
    data_path = "./dataset/finetuning/instruction-preference-data.json"
    pretrained_model_path = "./saved_results/finetuning_pretrained_models/"
    finetuned_model_name = "gpt2-medium355M-sft.pth"
    finetuned_model_path = os.path.join(pretrained_model_path, finetuned_model_name)
    # set seed
    torch.manual_seed(123)

    # data
    data = load_instruction_data(data_path = data_path)
    logger.info(f"Number of entries: {len(data)}")

    # dataset, dataloader
    (
        train_data, train_dataset, train_dataloader, 
        test_data, test_dataset, test_dataloader,
        valid_data, valid_dataset, valid_dataloader,
    ) = _build_data(data_path=data_path)

    logger.info(f"Train loader:")
    for batch in train_dataloader:
        logger.info(f"bathc['chosen'].shape: {batch['chosen'].shape}")
        logger.info(f"bathc['rejected'].shape: {batch['rejected'].shape}")
    
    # model
    model, base_config = _build_model(model_path=finetuned_model_path)

    # model test before dpo
    from layers.tokenization import text_to_token_ids, token_ids_to_text
    from models.gpt_generate import generate
    prompt = """Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    Convert the active sentence to passive: 'The chef cooks the meal every day.'
    """
    token_ids = generate(
        model = model,
        token_idx = text_to_token_ids(prompt),
        max_new_tokens = 35,
        context_size = base_config.context_length,
        eos_id = 50256,
    )
    response = token_ids_to_text(token_ids)
    logger.info(f"response: \n{response}") 


if __name__ == "__main__":
    main()
