CUDA_VISIBLE_DEVICES=0


# ----------------------
# Download Assets
# ----------------------
# Download tokenizer
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/tok4096.model -P data/

# Download pre-trained checkpoint
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/ckpt-v1.pt -P out/


# Run Inference
python sample.py \
    --prompt "Once upon a time" \
    --tokenizer_path data/tok4096.model \
    --ckpt_dir out/ \
    --num_samples 3 \
    --max_new_tokens 200 \
    --temperature 0.7
