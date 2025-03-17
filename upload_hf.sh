#!/bin/bash

CHECKPOINT_DIR="/export/work/sabreu/flame/exp/hgrn-sparse-340M-10B/batch32.gpu4.sparse90.steps20480.lr3e-4/checkpoint"
OUTPUT_PATH="/export/work/sabreu/flame/exp/hgrn-sparse-340M-10B/batch32.gpu4.sparse90.steps20480.lr3e-4/checkpoint/HF"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

echo "Converting checkpoint to Hugging Face format..."
python convert_dcp_to_hf.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --path "$OUTPUT_PATH" \
    --config configs/hgrn_s90_340M.json \
    --tokenizer fla-hub/transformer-1.3B-100B \
    --hf_name stevenabreu7/hgrn-sparse90-340M-5B

echo "Conversion and upload completed successfully!"
