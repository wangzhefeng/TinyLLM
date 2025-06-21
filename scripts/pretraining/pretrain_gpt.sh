export CUDA_VISIBLE_DEVICES=0


python -u ./model_pretraining/run_gpt_pretrain.py \
    --task_name tiny_gpt_pretrain \
    --model_name gpt \
    --is_train 1 \
    --is_test 0 \
    --is_inference 0 \
    --data_path ./dataset/pretrain/gpt/the-verdict.txt \
    --data "the-vardict.txt" \
    --context_length 256 \
    --vocab_size 50257 \
    --emb_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.1 \
    --qkv_bias 0 \
    --dtype torch.float32 \
    --max_new_tokens 50 \
    --tokenizer_model gpt2 \
    --seed 123 \
    --iters 1 \
    --train_epochs 10 \
    --batch_size 2 \
    --train_ratio 0.9 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --initial_lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --patience 7 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --num_workers 0 \
    --use_amp 1 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_multi_gpu 0 \
    --devices 0,1,2,3
