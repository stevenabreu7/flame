# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from fla.pruning import fuse_pruning_masks
from torchtitan.tools.logging import init_logger, logger


@torch.inference_mode()
def save_pretrained(
    checkpoint: str,
    path: str,
    config: str,
    tokenizer: str,
    hf_name: str,
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path}")
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        # fuse pruning masks, if possible
        fuse_pruning_masks(model)

        logger.info(f"Saving the model to {path}")
        model.save_pretrained(path)
        if hf_name:
            logger.info(f"Pushing the model to {hf_name}")
            model.push_to_hub(hf_name)

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--hf_name", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.path, ".metadata")):
        steps_folders = os.listdir(args.checkpoint)
        steps_folders = [f for f in steps_folders if f.startswith("step-")]
        steps_folders = sorted(steps_folders, key=lambda x: int(x.split("-")[-1]))
        latest_step = steps_folders[-1]
        print(f"Found folders: {steps_folders}")
        print(f"Latest step: {latest_step}")
        args.checkpoint = os.path.join(args.checkpoint, latest_step)

    save_pretrained(args.checkpoint, args.path, args.config, args.tokenizer, args.hf_name)
