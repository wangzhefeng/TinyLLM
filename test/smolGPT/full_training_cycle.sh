CUDA_VISIBLE_DEVICES=0

# 1.Prepare Dataset
python preprocess.py prepare-dataset --vocab-size 4096

# 2.Start Training
python train.py

# 3.Generate Text
python sample.py \
    --prompt "Once upon a time" \
    --num_samples 3 \
    --temperature 0.7 \
    --max_new_tokens 500
