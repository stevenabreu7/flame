#!/bin/bash
# Helper script for applying quantization to a pre-trained model

# Default configuration
MODEL_PATH=""
OUTPUT_PATH=""
MODEL_CONFIG=""
MODE="dynamic"
WEIGHT_FORMAT="INT8"
ACTIVATION_FORMAT="INT8"
WEIGHT_SYMMETRIC=true
ACTIVATION_SYMMETRIC=true
POWER_OF_TWO_SCALES=true
CALIBRATION_DATASET=""
CALIBRATION_BATCH_SIZE=16
CALIBRATION_NUM_BATCHES=10

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --model_config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --weight_format)
      WEIGHT_FORMAT="$2"
      shift 2
      ;;
    --activation_format)
      ACTIVATION_FORMAT="$2"
      shift 2
      ;;
    --weight_symmetric)
      WEIGHT_SYMMETRIC="$2"
      shift 2
      ;;
    --activation_symmetric)
      ACTIVATION_SYMMETRIC="$2"
      shift 2
      ;;
    --power_of_two_scales)
      POWER_OF_TWO_SCALES="$2"
      shift 2
      ;;
    --calibration_dataset)
      CALIBRATION_DATASET="$2"
      shift 2
      ;;
    --calibration_batch_size)
      CALIBRATION_BATCH_SIZE="$2"
      shift 2
      ;;
    --calibration_num_batches)
      CALIBRATION_NUM_BATCHES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model_path is required"
  exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
  echo "Error: --output_path is required"
  exit 1
fi

# Convert boolean arguments to flags
WEIGHT_SYMMETRIC_FLAG=""
if [ "$WEIGHT_SYMMETRIC" = "true" ]; then
  WEIGHT_SYMMETRIC_FLAG="--weight_symmetric"
fi

ACTIVATION_SYMMETRIC_FLAG=""
if [ "$ACTIVATION_SYMMETRIC" = "true" ]; then
  ACTIVATION_SYMMETRIC_FLAG="--activation_symmetric"
fi

POWER_OF_TWO_SCALES_FLAG=""
if [ "$POWER_OF_TWO_SCALES" = "true" ]; then
  POWER_OF_TWO_SCALES_FLAG="--power_of_two_scales"
fi

# Build the command
CMD="python quantize_model.py \
  --model_path $MODEL_PATH \
  --output_path $OUTPUT_PATH \
  --mode $MODE \
  --weight_format $WEIGHT_FORMAT \
  --activation_format $ACTIVATION_FORMAT"

# Add optional arguments if provided
if [ ! -z "$MODEL_CONFIG" ]; then
  CMD="$CMD --model_config $MODEL_CONFIG"
fi

# Add boolean flags
CMD="$CMD $WEIGHT_SYMMETRIC_FLAG $ACTIVATION_SYMMETRIC_FLAG $POWER_OF_TWO_SCALES_FLAG"

# Add calibration arguments if static quantization
if [ "$MODE" = "static" ]; then
  if [ ! -z "$CALIBRATION_DATASET" ]; then
    CMD="$CMD --calibration_dataset $CALIBRATION_DATASET"
  fi
  CMD="$CMD --calibration_batch_size $CALIBRATION_BATCH_SIZE --calibration_num_batches $CALIBRATION_NUM_BATCHES"
fi

# Print and execute the command
echo "Executing: $CMD"
eval "$CMD" 
