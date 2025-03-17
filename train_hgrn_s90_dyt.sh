#!/bin/bash
#SBATCH --job-name=hgrn_s90_dyt_train
#SBATCH --output=%x_%j.out
#SBATCH --partition=g80
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32

## Notes:
# - This script is based on train_hgrn_s90.sh but uses DyT normalization
# - It processes the model with dynamic_tanh_norm.py post-initialization

batchsize=32
alpha_init=0.5  # Initial value for DyT alpha parameter

# Add debugging output
echo "Starting job at $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"

# Fix pyenv rehash issue before sourcing
if [ -f "/home/sabreu/.pyenv/shims/.pyenv-shim" ]; then
    echo "Removing stale pyenv shim file"
    rm -f /home/sabreu/.pyenv/shims/.pyenv-shim
fi

# Activate virtual environment
source venv/bin/activate
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

# Create a wrapper script for training that applies DyT conversion
cat > train_with_dyt.py << 'EOF'
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Add the local path to the python path to import our custom module
sys.path.insert(0, os.path.abspath('3rdparty/flash-linear-attention'))

from fla.dynamic_tanh_norm import convert_rmsnorm_to_dyt

# Get the alpha init value from the environment variable
alpha_init_value = float(os.environ.get('DYT_ALPHA_INIT', '0.5'))

# Create a monkey patch for AutoModelForCausalLM.from_config
original_from_config = AutoModelForCausalLM.from_config

def from_config_with_dyt_patch(config, **kwargs):
    print(f"Initializing model from config and applying DyT normalization (alpha={alpha_init_value})")
    
    # Create model with original method
    model = original_from_config(config, **kwargs)
    
    # Convert RMSNorm to DyT
    model = convert_rmsnorm_to_dyt(model, alpha_init_value)
    
    print("Model initialization with DyT normalization complete")
    return model

# Apply the monkey patch
AutoModelForCausalLM.from_config = from_config_with_dyt_patch

# Run the original train.py script
exec(open("train.py").read())
EOF

# Export the alpha init value for the custom script
export DYT_ALPHA_INIT=$alpha_init

NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file train.toml \
  --job.dump_folder /export/work/sabreu/flame/exp/hgrn-sparse-340M-10B/batch${batchsize}.gpu4.sparse90.steps20480.lr3e-4.dyt \
  --model.config configs/hgrn_s90_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.lr 3e-4 \
  --optimizer.min_lr_ratio 0.1 \
  --optimizer.scheduler cosine \
  --training.batch_size ${batchsize} \
  --training.seq_len 2048 \
  --training.warmup_steps 1024 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 16 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 4 \
  -- python train_with_dyt.py