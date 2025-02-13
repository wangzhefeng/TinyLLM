
python training.py \ 
    # data params
    --data_source https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt \
    --train_ratio 0.9 \
    --context_length 1024 \ 
    --model_name gpt \  # model params
    --num_epochs 10 \  # training params
    --learning_rate 0.0004 \
    --weight_decay 0.1 \
    --user_gpu 1 \
    --gpu_type cuda \
    --user_multi_gpu 0 \
    --devices 0,1,2,3
    --checkpoints "./saved_results/pretrained_models/"

