export CUDA_VISIBLE_DEVICES=0


python -u ./model_pretraining/run_gpt_pretrain.py \
    --task_name tiny_gpt_pretrain \
    --is_training 1 \
    --is_inference 0 \
    --data_source https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt \
    --context_length 256 \
    --model_name gpt \
    --vocab_size 50257 \
    --emb_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.1 \
    --qkv_bias 0 \
    --max_new_tokens 50 \
    --iters 10 \
    --train_epochs 10 \
    --batch_size 2 \
    --train_ratio 0.9 \
    --learning_rate 0.0005 \
    --initial_lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --lradj type1 \
    --patience 7 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --use_amp 1 \
    --use_gpu 1 \
    --use_multi_gpu 0 \
    --gpu_type cuda \
    --devices 0,1,2,3
