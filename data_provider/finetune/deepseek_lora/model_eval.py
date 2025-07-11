# -*- coding: utf-8 -*-

# ***************************************************
# * File        : finetuning_eval.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-03-01
# * Version     : 0.1.030115
# * Description : https://mp.weixin.qq.com/s/ndNDMyipWJQzthRoJppIBw
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bert_score import score

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def generate_response(model, tokenizer, prompt):
    """
    统一的生成函数

    Args:
        model (_type_): 要使用的模型实例
        prompt (_type_): 符合格式要求的输入文本

    Returns:
        _type_: 清洗后的回答文本
    """
    # 输入编码（保持与训练时相同的处理方式）
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",  # 返回 PyTorch 张量
        max_length = 1024,  # 最大输入长度（与训练时一致）
        truncation = True,  # 启用截断
        padding = "max_length"  # 填充到最大长度（保证batch一致性）
    ).to(model.device)  # 确保输入与模型在同一设备
    # 文本生成（关闭梯度计算以节省内存）
    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs.input_ids,
            attention_mask = inputs.attention_mask,
            max_new_tokens = 1024,  # 生成内容的最大 token 数（控制回答长度）
            temperature = 0.7,  # 温度参数（0.0-1.0，值越大随机性越强）
            top_p = 0.9,  # 核采样参数（保留累积概率前 90% 的 token）
            repetition_penalty = 1.1,  # 重复惩罚系数（>1.0 时抑制重复内容）
            eos_token_id = tokenizer.eos_token_id,  # 结束符 ID
            pad_token_id = tokenizer.pad_token_id,  # 填充符 ID
        )
    # 解码与清洗输出
    full_text = tokenizer.decode(outputs[0], skip_special_tokens = True)  # 跳过特殊 token
    answer = full_text.split("### 答案：\n")[-1].strip()  # 提取答案部分

    return answer


def batch_generate_response(model, tokenizer, questions):
    """
    批量生成回答

    Args:
        model (_type_): _description_
        questions (_type_): _description_

    Returns:
        _type_: _description_
    """
    answers = []
    for q in tqdm(questions):
        prompt = f"诊断问题：{q}\n详细分析：\n### 答案：\n"
        ans = generate_response(model, tokenizer, prompt)
        answers.append(ans)
    
    return answers


def compare_models(base_model, lora_model, tokenizer, question):
    """
    模型对比函数
    
    Args:
        base_model (_type_): 原始预训练模型
        lora_model (_type_): LoRA 微调后的模型
        tokenizer (_type_): tokenizer
        question (_type_): 自然语言形式的医疗问题
    """
    # 构建符合训练格式的prompt（注意与训练时格式完全一致）
    prompt = f"诊断问题：{question}\n详细分析：\n### 答案：\n"
    # 原始模型
    base_answer = generate_response(
        model = base_model,
        tokenizer = tokenizer, 
        prompt = prompt
    )
    # 微调模型
    lora_answer = generate_response(
        model = lora_model, 
        tokenizer = tokenizer, 
        prompt = prompt
    )
    # 终端彩色打印对比结果
    logger.info("\n" + "="*50)
    logger.info(f"问题：{question}")
    logger.info("-"*50)
    logger.info(f"\033[1;34m[原始模型]\033[0m\n{base_answer}")  # 蓝色显示原始模型结果
    logger.info("-"*50)
    logger.info(f"\033[1;32m[LoRA模型]\033[0m\n{lora_answer}")  # 绿色显示微调模型结果
    logger.info("="*50 + "\n")


def compare_model_bertscore(base_model, lora_model, tokenizer, test_data):
    # 生成结果
    base_answers = batch_generate_response(base_model, tokenizer, [d["Question"] for d in test_data])
    lora_answers = batch_generate_response(lora_model, tokenizer, [d["Question"] for d in test_data])
    ref_answers = [d["Response"] for d in test_data]
    # bert model path(https://huggingface.co/google-bert/bert-base-chinese)
    bert_model_path = "downloaded_models/bert-base-chinese"
    # 计算 BERTScore
    _, _, base_bert = score(
        base_answers, 
        ref_answers, 
        lang = "zh",
        model_type = bert_model_path,
        num_layers = 12,
        device = "cuda"
    )
    _, _, lora_bert = score(
        lora_answers, 
        ref_answers, 
        lang="zh",
        model_type = bert_model_path,
        num_layers=12, 
        device = "cuda"
    )
    logger.info(f"BERTScore | 原始模型: {base_bert.mean().item():.3f} | LoRA模型: {lora_bert.mean().item():.3f}")


def load_models():
    """
    加载模型
    """
    # 原始预训练模型路径
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # LoRA 微调后保存的适配器路径
    peft_model_path = "saved_results/finetuned_models/medical_o1_sft_Chinese"

    # 初始化分词器（使用与训练时相同的 tokenizer）
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 加载基础模型（半精度加载节省显存）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype = torch.float16,
        device_map = "auto",
    )

    # 加载 LoRA 适配器（在基础模型上加载微调参数）
    lora_model = PeftModel.from_pretrained(
        model = base_model, 
        model_id = peft_model_path,
        torch_dtype = torch.float16,
        device_map = "auto",
    )

    # 合并 LoRA 权重到基础模型（提升推理速度，但会失去再次训练的能力）
    lora_model = lora_model.merge_and_unload()

    # 设置为评估模式
    lora_model.eval()

    return base_model, lora_model, tokenizer




# 测试代码 main 函数
def main():
    # ------------------------------
    # 加载模型
    # ------------------------------
    base_model, lora_model, tokenizer = load_models()
    # ------------------------------
    # 
    # ------------------------------
    # 测试问题集（可自由扩展）
    test_questions = [
        "根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？"
    ]
    # 遍历测试问题
    for q in test_questions:
        compare_models(
            base_model = base_model, 
            lora_model = lora_model, 
            tokenizer = tokenizer,
            question = q, 
        )
    """
    # ------------------------------
    # 批量测试
    # ------------------------------
    # 加载测试数据
    data_path = "dataset/finetune/AI-ModelScope/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json"
    with open(data_path) as f:
        test_data = json.load(f) 
    # 数据量比较大，只选择 10 条数据进行测试
    test_data = test_data[:10]
    
    # 遍历测试问题
    compare_model_bertscore(
        base_model = base_model, 
        lora_model = lora_model, 
        tokenizer = tokenizer, 
        test_data = test_data
    )
    """

if __name__ == "__main__":
    main()
