export CUDA_VISIBLE_DEVICES="0"


python -u ./model_finetuning/run_gpt_instruction_sft.py \
    --task_name tiny_gpt_instruction_sft \
    --model_name gpt_finetune_instruction \
    --is_train 0 \
    --is_test 1 \
    --is_eval 0 \
    --is_inference 1 \
    --data_path ./dataset/finetune/instruction-data.json \
    --context_length 1024 \
    --train_ratio 0.85 \
    --test_ratio 0.10 \
    --vocab_size 50257 \
    --emb_dim 768 \
    --n_layers 12 \
    --n_heads 12 \
    --dropout 0.0 \
    --qkv_bias 1 \
    --pretrained_model 'gpt2-medium (355M)' \
    --pretrained_model_path ./downloaded_models/gpt2_model \
    --pretrained_model_source huggingface_gpt2 \
    --finetuned_model_path ./saved_results/finetuned_models \
    --tokenizer_model gpt2 \
    --seed 123 \
    --iters 1 \
    --train_epochs 10 \
    --max_new_tokens 256 \
    --batch_size 8 \
    --learning_rate 0.00005 \
    --weight_decay 0.1 \
    --checkpoints ./saved_results/finetuned_models \
    --test_results ./saved_results/test_results \
    --eval_data_path ./dataset/finetune/instruction-data-with-response.json \
    --num_workers 0 \
    --use_gpu 1 \
    --use_multi_gpu 0 \
    --gpu_type cuda \
    --devices 0,1,2,3
