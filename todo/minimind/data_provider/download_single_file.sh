pip install modelscope

modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir ./dataset/minimind
modelscope download --dataset gongjy/minimind_dataset sft_mini_512.jsonl --local_dir .dataset/minimind
