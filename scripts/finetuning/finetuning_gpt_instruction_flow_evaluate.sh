export CUDA_VISIBLE_DEVICES="0"


python -u ./model_finetuning/run_gpt_instruction_sft_evaluate.py \
    --task_name tiny_gpt_instruction_sft_evaluate \
    --model_name gpt_instruction_sft \
    --data_path ./dataset/finetune/instruction-data-with-response.json \
    --train_ratio 0.85 \
    --test_ratio 0.10 \
    --inference_server ollama \
    --inference_server_url http://localhost:11434/api/chat \
    --inference_model  llama3 \
    --num_ctx 2048 \
    --iter 1 \
    --seed 123 \
    --num_workers 0 \
    --use_gpu 1 \
    --use_multi_gpu 0 \
    --gpu_type cuda \
    --devices 0,1,2,3
