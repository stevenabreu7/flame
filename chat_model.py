#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simple script to load a Flame model from checkpoint and chat with it

import argparse
import io
import os
import sys
from datetime import timedelta

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger

def setup_model_and_tokenizer(checkpoint_dir, config_path, tokenizer_path, device='cuda'):
    """
    Load a model and tokenizer from checkpoint.
    
    Args:
        checkpoint_dir: Path to the directory containing the checkpoint
        config_path: Path to the model config file
        tokenizer_path: Path to the tokenizer
        device: Device to load the model onto ('cuda' or 'cpu')
        
    Returns:
        model, tokenizer
    """
    logger.info(f"Loading config from {config_path}")
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Find the latest checkpoint
    logger.info(f"Looking for checkpoints in {checkpoint_dir}")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('step_')]
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
    # Sort by step number to get the latest checkpoint
    latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Create a temporary file to convert from DCP format to standard PyTorch
    with torch.inference_mode():
        with torch.device('cpu'):
            # Allow timedelta and BytesIO in torch.load
            torch.serialization.add_safe_globals([timedelta, io.BytesIO])
            
            # Convert DCP to regular PyTorch checkpoint
            logger.info("Converting DCP checkpoint to PyTorch format")
            with torch.inference_mode(), torch.device('cpu'):
                temp_checkpoint = os.path.join(os.path.dirname(checkpoint_dir), 'temp_checkpoint.pt')
                dcp_to_torch_save(checkpoint_path, temp_checkpoint)
                
                # Create model from config
                logger.info("Initializing model from config")
                model = AutoModelForCausalLM.from_config(config)
                
                # Load state dict
                logger.info("Loading state dict from checkpoint")
                state_dict = torch.load(temp_checkpoint, map_location='cpu')
                model.load_state_dict(state_dict['model'])
                
                # Clean up the temporary file
                os.remove(temp_checkpoint)
    
    # Move to the appropriate device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
        logger.info("Model loaded on GPU")
    else:
        logger.info("Model loaded on CPU")
    
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.95, do_sample=True):
    """
    Generate text from a prompt.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: The input prompt
        max_length: Maximum length of the generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling (vs greedy decoding)
        
    Returns:
        Generated text
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Move to the same device as the model
    if next(model.parameters()).is_cuda:
        input_ids = input_ids.to('cuda')
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def chat_loop(model, tokenizer):
    """
    Simple chat loop to interact with the model.
    """
    print("\nWelcome to the Flame chat interface!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    conversation_history = ""
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # Add user input to conversation history
        conversation_history += f"\nUser: {user_input}\nAssistant: "
        
        # Generate response
        generated_text = generate_text(
            model, 
            tokenizer, 
            conversation_history, 
            max_length=len(conversation_history) + 200
        )
        
        # Extract just the assistant's response
        assistant_response = generated_text.split("Assistant: ")[-1]
        
        # Update conversation history
        conversation_history = generated_text
        
        print(f"\nAssistant: {assistant_response}")

def main():
    parser = argparse.ArgumentParser(description="Chat with a Flame model loaded from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the model config file")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to the tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional prompt to generate from (if not using chat mode)")
    
    args = parser.parse_args()
    
    init_logger()
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.checkpoint, 
        args.config, 
        args.tokenizer, 
        args.device
    )
    
    if args.prompt:
        # Generate from a single prompt
        generated_text = generate_text(model, tokenizer, args.prompt)
        print(f"\nGenerated text:\n{generated_text}")
    else:
        # Enter interactive chat mode
        chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()