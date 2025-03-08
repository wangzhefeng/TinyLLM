export CUDA_VISIBLE_DEVICES="0"


python -u run_gpt_clf_sft.py \
    --task_name tiny_gpt_classification_sft \
    --model_name gpt_finetune_clf \
    --is_training 1 \
    --is_inference 0 \
    --data_source ./dataset/finetune/sms_spam_collection \
    --context_length 1024 \
    --num_classes 2 \
    --vocab_size 50257 \
    --emb_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.0 \
    --qkv_bias 1 \
    --model_path ./saved_results/pretrained_models/tiny_gpt_pretrain_gpt_the-verdict_cl256_te10_bs2/checkpoint.pth \
    --pretrained_model 'gpt2-small (124M)' \
    --pretrained_model_path ./downloaded_models/gpt2_model \
    --pretrained_model_source huggingface_gpt2 \
    --finetune_method simple \
    --tokenizer_model gpt2 \
    --seed 123 \
    --iters 10 \
    --train_epochs 5 \
    --batch_size 8 \
    --num_workers 0 \
    --learning_rate 0.00005 \
    --weight_decay 0.1 \
    --use_gpu 1 \
    --use_multi_gpu 0 \
    --gpu_type cuda \
    --devices 0,1,2,3
