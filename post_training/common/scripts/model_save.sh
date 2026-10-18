
python scripts/merge_adapter_weights.py \
    --peft_model_id runs/llama-3-1-8b-math-orca-qlora-10k-ep1 \
    --push_to_hub True \
    --repository_id llama-3-1-8b-math-orca-qlora-10k-ep1-merged
