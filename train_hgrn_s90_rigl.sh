#!/bin/bash
#SBATCH --job-name=hgrn_s90_rigl_train
#SBATCH --output=%x_%j.out
#SBATCH --partition=g80
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32

## Notes:
# - This script uses RigL dynamic sparse training (vs static magnitude pruning)
# - The configuration uses more frequent topology updates (100 steps vs 256)
# - Dynamic topology updates continue for 90% of training vs 50% in magnitude pruning

batchsize=32

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

NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file train.toml \
  --job.dump_folder /export/work/sabreu/flame/exp/hgrn-sparse-340M-10B/batch${batchsize}.gpu4.sparse90-rigl.steps20480.lr3e-4 \
  --model.config configs/hgrn_s90_340M_rigl.json \
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
  --metrics.log_freq 4