# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility script to apply quantization to a trained model for efficient inference.
This script supports both dynamic and static quantization.
"""

import argparse
import os
import sys
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from flame.quantization import QuantFormat, QuantizationConfig, QuantizedLinear
from torchtitan.tools.logging import init_logger, logger


class QuantizationMode(Enum):
    """Quantization mode for applying quantization to a trained model."""
    DYNAMIC = "dynamic"  # Activations are quantized on-the-fly during inference
    STATIC = "static"    # Activations are pre-quantized based on calibration data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Apply quantization to a trained model")
    
    # Model arguments
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained model or checkpoint to quantize"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save the quantized model"
    )
    parser.add_argument(
        "--model_config", type=str, default=None,
        help="Path to the model configuration file"
    )
    
    # Quantization arguments
    parser.add_argument(
        "--mode", type=str, choices=["dynamic", "static"], default="dynamic",
        help="Quantization mode: dynamic or static"
    )
    parser.add_argument(
        "--weight_format", type=str, 
        choices=["INT4", "INT8", "FLOAT8"], default="INT8",
        help="Quantization format for weights"
    )
    parser.add_argument(
        "--activation_format", type=str,
        choices=["INT4", "INT8", "FLOAT8"], default="INT8",
        help="Quantization format for activations"
    )
    parser.add_argument(
        "--weight_symmetric", action="store_true", default=True,
        help="Use symmetric quantization for weights"
    )
    parser.add_argument(
        "--activation_symmetric", action="store_true", default=True,
        help="Use symmetric quantization for activations"
    )
    parser.add_argument(
        "--power_of_two_scales", action="store_true", default=True,
        help="Use powers of two for quantization scales"
    )
    
    # Calibration arguments (for static quantization)
    parser.add_argument(
        "--calibration_dataset", type=str, default=None,
        help="Path to the calibration dataset (for static quantization)"
    )
    parser.add_argument(
        "--calibration_batch_size", type=int, default=16,
        help="Batch size for calibration (for static quantization)"
    )
    parser.add_argument(
        "--calibration_num_batches", type=int, default=10,
        help="Number of batches to use for calibration (for static quantization)"
    )
    
    return parser.parse_args()


def apply_quantization_to_linear_module(
    module: nn.Module,
    config: QuantizationConfig,
) -> None:
    """
    Apply quantization to linear modules by replacing them with quantized versions.
    
    Args:
        module: PyTorch module to quantize
        config: Quantization configuration
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # Replace with quantized linear module
            logger.info(f"Quantizing linear layer: {name}")
            setattr(module, name, QuantizedLinear(child, config))
        else:
            # Recursively apply to child modules
            apply_quantization_to_linear_module(child, config)


def prepare_calibration_loader(
    args: argparse.Namespace,
) -> Optional[torch.utils.data.DataLoader]:
    """
    Prepare dataloader for calibration (static quantization).
    
    Args:
        args: Command line arguments
        
    Returns:
        DataLoader for calibration or None if no calibration is needed
    """
    if args.mode != "static" or not args.calibration_dataset:
        return None
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        logger.info(f"Loading calibration dataset: {args.calibration_dataset}")
        dataset = load_dataset(args.calibration_dataset, split="train")
        
        # Assuming the model was trained using the same tokenizer
        # We need to load the tokenizer used during training
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"], 
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
        tokenized_dataset = tokenized_dataset.select(
            range(min(len(tokenized_dataset), args.calibration_batch_size * args.calibration_num_batches))
        )
        
        calibration_dataloader = torch.utils.data.DataLoader(
            tokenized_dataset, 
            batch_size=args.calibration_batch_size
        )
        
        return calibration_dataloader
    except ImportError:
        logger.warning("Could not import datasets and/or transformers. Skipping calibration.")
        return None


def calibrate_model(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """
    Calibrate the model for static quantization.
    
    Args:
        model: Model to calibrate
        calibration_loader: DataLoader with calibration data
        device: Device to run calibration on
    """
    logger.info("Starting model calibration...")
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            # Move input to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass to collect activation statistics
            # Note: The QuantizedLinear module will collect statistics during forward pass
            model(input_ids, attention_mask=attention_mask)
            
            logger.info(f"Calibration: processed batch {batch_idx + 1}/{len(calibration_loader)}")
            
            # Limit calibration to the specified number of batches
            if batch_idx + 1 >= args.calibration_num_batches:
                break
    
    logger.info("Calibration completed")


def main(args: argparse.Namespace) -> None:
    """
    Main function to apply quantization to a trained model.
    
    Args:
        args: Command line arguments
    """
    # Initialize logger
    init_logger()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create quantization config
    quant_config = QuantizationConfig(
        weight_format=QuantFormat[args.weight_format],
        activation_format=QuantFormat[args.activation_format],
        weight_symmetric=args.weight_symmetric,
        activation_symmetric=args.activation_symmetric,
        static_quantization=(args.mode == "static"),
        power_of_two_scales=args.power_of_two_scales,
    )
    
    # Load model config if provided, otherwise use the one from the model path
    model_config_path = args.model_config or args.model_path
    
    # Load the model
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,  # Use float16 for efficiency
        config=AutoConfig.from_pretrained(model_config_path) if args.model_config else None,
    )
    
    # Move model to device
    model = model.to(device)
    
    # Apply quantization to the model
    logger.info(f"Applying {args.mode} quantization with {args.weight_format} weights and {args.activation_format} activations")
    apply_quantization_to_linear_module(model, quant_config)
    
    # Calibrate model if using static quantization
    if args.mode == "static":
        calibration_loader = prepare_calibration_loader(args)
        if calibration_loader:
            calibrate_model(model, calibration_loader, device)
        else:
            logger.warning("No calibration performed for static quantization. Results may be suboptimal.")
    
    # Save the quantized model
    logger.info(f"Saving quantized model to {args.output_path}")
    model.save_pretrained(args.output_path)
    
    logger.info("Quantization completed successfully")


if __name__ == "__main__":
    args = parse_args()
    main(args) 
