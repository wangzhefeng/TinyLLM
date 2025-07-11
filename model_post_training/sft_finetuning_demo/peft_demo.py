# -*- coding: utf-8 -*-

# ***************************************************
# * File        : peft_demo.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 0.1.103113
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


from peft import LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# ------------------------------
# peft config
# ------------------------------
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

# ------------------------------
# base model
# ------------------------------
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")

# ------------------------------
# peft model 
# ------------------------------
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ------------------------------
# train model
# ------------------------------
training_args = TrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ------------------------------
# save model
# ------------------------------
# save model to local
model.save_pretrained("output_dir")


# save model to huggingface hub
from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("your-name/bigscience/mt0-large-lora")

# ------------------------------
# inference
# ------------------------------
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model = model.to("cuda")
model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

# or
from peft import AutoPeftModel

model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
