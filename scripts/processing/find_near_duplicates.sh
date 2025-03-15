
python -u ./data_provider/finetune/instruction_follow/find_near_duplicates.py \
    --json_file ./dataset/finetune/instruction-examples.json \
    --threshold 0.9 \
    --remove_duplicates 1 \
    --json_output_file ./dataset/finetune/instruction-examples-without-duplicates.json
