export CUDA_VISIBLE_DEVICES="0"


python -u run_gpt_classification_sft.py \
    --task_name tiny_gpt_classification_sft \
    --is_training 1 \
    --is_inference 0 \
    --data_source ./dataset\finetuning\sms_spam_collection \
    --batch_size 8 \
    --vocab_size 50257 \
    --context_length 1024 \
    --dropout 0.0 \
    --qkv_bias 1 \
    --pretrained_model 'gpt2-small (124)' \
    --pretrained_model_path ./downloaded_models/gpt2_model \
    --num_classes 2 \
    --train_epochs 10 \
    --use_gpu 1 \
    --use_multi_gpu 0 \
    --gpu_type cuda \
    --devices 0,1,2,3
