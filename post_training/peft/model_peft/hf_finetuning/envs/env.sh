platform=$(uname -s)
echo $platform


# Env setup
# ---------------
if [ $platform == "Darwin" ]; then
    # Install PyTorch & other libraries
    pip install "torch==2.1.2" tensorboard

    # Install HuggingFace libraries
    pip install --upgrade \
        "transformers==4.36.2" \
        "datasets==2.16.1" \
        "accelerate==0.26.1" \
        "evaluate==0.4.1" \
        "bitsandbytes==0.42.0" \
        "trl==0.7.10" \
        "peft==0.7.1" \

    # Install peft & trl from github
    pip install git+https://github.com/huggingface/trl@a3c5b7178ac4f65569975efadc97db2f3749c65e --upgrade
    pip install git+https://github.com/huggingface/peft@4a1559582281fc3c9283892caea8ccef1d6f5a4f --upgrade
elif [ $platform == "MSYS_NT-10.0-26100" ]; then
    # Install PyTorch & other libraries
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    uv pip install tensorboard

    # Install HuggingFace libraries
    uv pip install --upgrade \
        transformers \
        datasets \
        accelerate \
        evaluate \
        bitsandbytes \
        trl \
        peft \
        lighteval \
        hf-transfer
    
    # Install peft & trl from github
    # uv pip install git+https://github.com/huggingface/trl@a3c5b7178ac4f65569975efadc97db2f3749c65e --upgrade
    # uv pip install git+https://github.com/huggingface/peft@4a1559582281fc3c9283892caea8ccef1d6f5a4f --upgrade
    
    # Install flash-attn(install need 10~45 minutes)
    uv pip install ninja packaging
    # MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
fi
