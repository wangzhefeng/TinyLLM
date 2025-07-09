# -*- coding: utf-8 -*-

# ***************************************************
# * File        : peft_.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-25
# * Version     : 1.0.022514
# * Description : article:https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg==&mid=2247701892&idx=1&sn=c324f7689c7e5c9d8cf320f2b65e772a&chksm=e87c04c9df0b8ddf77806278042b364e7cda85081b0e6d6092a2b4b049671dcddca2fff29eed&scene=178&cur_album_id=3843452730144161794#rd
# *               dataset: https://modelscope.cn/datasets/AI-ModelScope/medical-o1-reasoning-SFT/quickstart
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from utils.device import device_setting
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# device
device = device_setting()


class LossCallback(TrainerCallback):
    """
    自定义回调记录 Loss
    """
    
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])


def process_data(tokenizer, data_path):
    """
    数据预处理函数

    instruction:
        诊断问题：
        详细分析：
        ### 答案：<|endoftext|>
    """
    # 数据加载
    dataset = load_dataset("json", data_files = data_path, split = "train[:1500]")
    # 数据格式化
    def format_example(example):
        instruction = f"诊断问题：{example['Question']}\n详细分析：{example['Complex_CoT']}"
        inputs = tokenizer(
            f"{instruction}\n### 答案：\n{example['Response']}<|endoftext|>",
            padding = "max_length",
            truncation = True,
            max_length = 512,
            return_tensors = "pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0), 
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

    return dataset.map(format_example, remove_columns=dataset.column_names)


def data_collator(data):
    """
    数据加载器
    """
    batch = {
        "input_ids": torch.stack([
            torch.tensor(d["input_ids"]) for d in data
        ]).to(device),
        "attention_mask": torch.stack([
            torch.tensor(d["attention_mask"]) for d in data
        ]).to(device),
        "labels": torch.stack([
            torch.tensor(d["input_ids"]) for d in data
        ]).to(device)  # 使用 input_ids 作为 labels
    }
    
    return batch


def plot_losses(loss_callback, output_path):
    """
    绘制训练集损失Loss曲线
    """
    plt.figure(figsize = (10, 6))
    plt.plot(loss_callback.losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(Path(output_path).joinpath("loss_curve.png"))


def finetuning(base_model_path, data_path, output_path):
    # 加载基础模型 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype = torch.float16,
        device_map = {"": device}  # 强制使用指定 GPU
    )
    logger.info(f"model: \n{model}")

    # LoRA 配置
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 32,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSAL_LM"
    )
    # LoRA model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据
    dataset = process_data(tokenizer, data_path)

    # 训练回调
    loss_callback = LossCallback()
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir = output_path,
        per_device_train_batch_size = 2,  # 显存优化设置
        gradient_accumulation_steps = 4,  # 累计梯度相当于 batch_size=8
        num_train_epochs = 3,
        learning_rate = 3e-4,
        fp16 = True,  # 开启混合精度
        logging_steps = 20,
        save_strategy = "no",
        report_to = "none",
        optim = "adamw_torch",
        no_cuda = False,  # 强制使用 CUDA
        dataloader_pin_memory = False,  # 加速数据加载
        remove_unused_columns = False,  # 防止删除未使用的列
    )
    
    # 创建 Trainer 
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = dataset,
        data_collator = data_collator,
        callbacks = [loss_callback]
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存最终模型
    trainer.model.save_pretrained(output_path)
    logger.info(f"模型已保存至：{output_path}")

    # 绘制训练集损失Loss曲线
    plot_losses(loss_callback, output_path)
    logger.info(f"Loss 曲线已保存至: {output_path}")




# 测试代码 main 函数
def main():
    # 强制使用 GPU
    assert torch.cuda.is_available(), "必须使用 GPU 进行训练！"

    # 基础模型路径
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # 数据集路径
    data_path = "dataset/finetune/AI-ModelScope/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json"
    # 微调后模型保存路径
    output_path = "saved_results/finetuned_models/medical_o1_sft_Chinese"
    os.makedirs(output_path, exist_ok=True)

    # 模型微调
    finetuning(
        base_model_path,
        data_path,
        output_path,
    )

if __name__ == "__main__":
    main()
