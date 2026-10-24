
python -u ./post_training/sft/data_provider/instruction_follow/find_near_duplicates.py \
    --json_file ./dataset/finetuning/instruction-examples.json \
    --threshold 0.9 \
    --remove_duplicates 1 \
    --json_output_file ./dataset/finetuning/instruction-examples-without-duplicates.json
