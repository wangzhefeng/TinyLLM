export CUDA_VISIBLE_DEVICES="0"


python -u ./model_finetuning/run_gpt_clf_sft.py \
    --task_name tiny_gpt_classification_sft \
    --model_name gpt_finetune_clf \
    --is_train 1 \
    --is_inference 1 \
    --data_path ./dataset/finetune/sms_spam_collection \
    --context_length 1024 \
    --num_classes 2 \
    --vocab_size 50257 \
    --emb_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.0 \
    --qkv_bias 1 \
    --pretrained_model 'gpt2-small (124M)' \
    --pretrained_model_path ./downloaded_models/gpt2_model \
    --pretrained_model_source huggingface_gpt2 \
    --finetuned_model_path ./saved_results/finetuned_models \
    --finetune_method simple \
    --tokenizer_model gpt2 \
    --seed 123 \
    --iters 1 \
    --train_epochs 5 \
    --batch_size 8 \
    --learning_rate 0.00005 \
    --lradj type1 \
    --weight_decay 0.1 \
    --checkpoints ./saved_results/finetuned_models \
    --test_results ./saved_results/test_results \
    --num_workers 0 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_multi_gpu 0 \
    --devices 0,1,2,3
