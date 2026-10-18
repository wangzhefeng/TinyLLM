export CUDA_VISIBLE_DEVICES="0"


python -u ./model_finetuning/run_gpt_preference_sft.py \
    --task_name tiny_gpt_instruction_sft \
    --is_train 1 \
    --is_inference 0 \
    --data_path ./dataset/finetune/instruction-preference-data.json \
    --train_ratio 0.85 \
    --test_ratio 0.10 \
    --batch_size 2 \
    --vocab_size 50257 \
    --context_length 1024 \
    --dropout 0.0 \
    --qkv_bias 1 \
    --pretrained_model 'gpt2-medium (355M)' \
    --pretrained_model_path ./downloaded_models/gpt2_model \
    --tokenizer_model "gpt2" \
    --finetuned_model_path ./saved_results/finetuned_models/gpt2-medium355M-sft.pth \
    --num_classes 2 \
    --train_epochs 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --use_gpu 1 \
    --use_multi_gpu 0 \
    --gpu_type cuda \
    --devices 0,1,2,3
