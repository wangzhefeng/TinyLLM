# clone repo
# git clone https://github.com/huggingface/nanoVLM.git
# cd nanoVLM

# env
cd ~
mkdir nanoVLM
cd nanoVLM
uv init --bare --python 3.12
uv sync --python 3.12
uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb

# training
wandb login --relogin
huggingface-cli login
python train.py

# generate
python generate.py


# evaluate
# Install lmms-eval (has to be from source)
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

# Make sure you have your environment variables set correctly and you are logged in to HF
export HF_HOME="<Path to HF cache>"
huggingface-cli login

# Evaluate a trained model on multiple benchmarks
python evaluation.py --model lusxvr/nanoVLM-450M --tasks mmstar,mme
