#!/bin/bash
#SBATCH --job-name=hgrn_s90_quantized_train
#SBATCH --output=%x_%j.out
#SBATCH --partition=g80
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32

## Notes:
# - This script is based on train_hgrn_s90.sh but includes options for quantization-aware training
# - Default configuration is dynamic INT8 quantization for both weights and activations
# - Scales are shared per-tensor and are powers of two

batchsize=32
weight_format=INT8        # Options: INT4, INT8, FLOAT8
activation_format=INT8    # Options: INT4, INT8, FLOAT8
weight_symmetric=true     # Whether to use symmetric quantization for weights
activation_symmetric=true # Whether to use symmetric quantization for activations
static_quantization=false # Whether to use static quantization (default is dynamic)
power_of_two_scales=true  # Whether scales should be powers of two

# Convert boolean values to lowercase strings for the TOML config
weight_symmetric_val=$(echo "$weight_symmetric" | tr '[:upper:]' '[:lower:]')
activation_symmetric_val=$(echo "$activation_symmetric" | tr '[:upper:]' '[:lower:]')
static_quantization_val=$(echo "$static_quantization" | tr '[:upper:]' '[:lower:]')
power_of_two_scales_val=$(echo "$power_of_two_scales" | tr '[:upper:]' '[:lower:]')

# Add debugging output
echo "Starting job at $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Quantization settings:"
echo "  - Weight format: $weight_format"
echo "  - Activation format: $activation_format"
echo "  - Weight symmetric: $weight_symmetric_val"
echo "  - Activation symmetric: $activation_symmetric_val"
echo "  - Static quantization: $static_quantization_val"
echo "  - Power of two scales: $power_of_two_scales_val"

# Fix pyenv rehash issue before sourcing
if [ -f "/home/sabreu/.pyenv/shims/.pyenv-shim" ]; then
    echo "Removing stale pyenv shim file"
    rm -f /home/sabreu/.pyenv/shims/.pyenv-shim
fi

# Activate virtual environment
source venv/bin/activate
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

# Create a quantization config file
cat > quantization_config.toml << EOF
[experimental.quantization]
weight_format = "$weight_format"
activation_format = "$activation_format"
weight_symmetric = $weight_symmetric_val
activation_symmetric = $activation_symmetric_val
static_quantization = $static_quantization_val
power_of_two_scales = $power_of_two_scales_val
EOF

# Run the training with quantization-aware training enabled
NNODE=1 NGPU=4 LOG_RANK=0 bash train.sh \
  --job.config_file train.toml \
  --job.dump_folder /export/work/sabreu/flame/exp/hgrn-sparse-340M-10B/batch${batchsize}.gpu4.sparse90.steps20480.lr3e-4.quantized_${weight_format}_${activation_format} \
  --model.config configs/hgrn_s90_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --model.converters quantization_aware_training \
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
  --quantization.config quantization_config.toml
