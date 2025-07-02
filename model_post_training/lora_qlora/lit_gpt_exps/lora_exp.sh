# 1.克隆仓库并安装依赖
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
pip install -r requirements.txt


# 2.下载并准备模型检查点
python scripts/download.py \
  --repo_id mistralai/Mistral-7B-Instruct-v0.1
# there are many other supported models
python scripts/convert_hf_checkpoint.py \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1


# 3.准备数据集
python scripts/prepare_alpaca.py \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1
# or from a custom CSV file
python scripts/prepare_csv.py \
  --csv_dir MyDataset.csv \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1


# 4.微调
python finetune/lora.py \
  --checkpoint_dir checkpoints/mistralai/Mistral-7B-Instruct-v0.1/ \
  --precision bf16-true


# 5.合并 LoRA 权重
python scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/mistralai/Mistral-7B-Instruct-v0.1" \
  --lora_path "out/lora/alpaca/Mistral-7B-Instruct-v0.1/lit_model_lora_finetuned.pth" \
  --out_dir "out/lora_merged/Mistral-7B-Instruct-v0.1/"

cp checkpoints/mistralai/Mistral-7B-Instruct-v0.1/*.json \
  out/lora_merged/Mistral-7B-Instruct-v0.1/


# 6.评估
python eval/lm_eval_harness.py \
  --checkpoint_dir "out/lora_merged/Mistral-7B-Instruct-v0.1/" \
  --eval_tasks "[arithmetic_2ds, ..., truthfulqa_mc]" \
  --precision "bf16-true" \
  --batch_size 4 \
  --num_fewshot 0 \
  --save_filepath "results.json"


# 7.使用
python chat/base.py \ 
  --checkpoint_dir "out/lora_merged/Mistral-7B-Instruct-v0.1/"