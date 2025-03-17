#!/bin/bash

# Script to load and chat with a trained HGRN-S90 model

# Set environment variables for better debugging
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export CUDA_LAUNCH_BLOCKING=1

# Get the most recent checkpoint directory
CHECKPOINT_DIR="/export/work/sabreu/flame/exp/hgrn-sparse-340M-10B/batch32.gpu4.sparse90.steps20480.lr3e-4/checkpoint"

# If you don't have CUDA available or want to force CPU
# DEVICE="cpu" 
DEVICE="cuda"

# Run the chat model script
python chat_model.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --config configs/hgrn_s90_340M.json \
  --tokenizer fla-hub/transformer-1.3B-100B \
  --prompt "The meaning of life is" \
  --temperature 0.0 \
  --fuse_mask \
  --device $DEVICE

python chat_model.py \
  --checkpoint "$CHECKPOINT_DIR" \
  --config configs/hgrn_s90_340M.json \
  --tokenizer fla-hub/transformer-1.3B-100B \
  --prompt "The meaning of life is" \
  --temperature 0.0 \
  --device $DEVICE

# Alternative: Generate from a prompt without entering chat loop
# python chat_model.py \
#   --checkpoint "$CHECKPOINT_DIR" \
#   --config configs/hgrn_s90_340M.json \
#   --tokenizer fla-hub/transformer-1.3B-100B \
#   --device $DEVICE \
#   --prompt "The meaning of life is"
