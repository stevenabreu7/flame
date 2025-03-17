# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization module for FLAME models.
Implements quantization-aware training with support for different quantization formats
for activations and weights, with per-tensor scales that are powers of two.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import _functional_collectives as funcol

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.tools.logging import logger


class QuantFormat(Enum):
    """Quantization formats supported."""
    INT4 = 4
    INT8 = 8
    FLOAT8 = 8 
    INT16 = 16
    INT24 = 24
    # We can add more formats later (e.g., FLOAT4, INT2, etc.)


class QuantizationConfig:
    """Configuration for quantization-aware training."""
    
    def __init__(
        self,
        weight_format: QuantFormat = QuantFormat.INT8,
        activation_format: QuantFormat = QuantFormat.INT8,
        weight_symmetric: bool = True,
        activation_symmetric: bool = True,
        static_quantization: bool = False,
        power_of_two_scales: bool = True,
    ):
        """
        Initialize quantization configuration.
        
        Args:
            weight_format: Quantization format for weights
            activation_format: Quantization format for activations
            weight_symmetric: Whether to use symmetric quantization for weights
            activation_symmetric: Whether to use symmetric quantization for activations
            static_quantization: Whether to use static quantization
            power_of_two_scales: Whether scales should be powers of two
        """
        self.weight_format = weight_format
        self.activation_format = activation_format
        self.weight_symmetric = weight_symmetric
        self.activation_symmetric = activation_symmetric
        self.static_quantization = static_quantization
        self.power_of_two_scales = power_of_two_scales


class QuantizedLinear(nn.Module):
    """
    Quantization-aware Linear layer that simulates quantization during training.
    
    This layer performs fake quantization of weights and activations during
    forward pass to simulate the effect of quantization, while keeping the
    weights and activations in floating point for backpropagation.
    """
    
    def __init__(
        self,
        linear_module: nn.Linear,
        config: QuantizationConfig,
    ):
        super().__init__()
        self.linear = linear_module
        self.config = config
        self.weight_bits = self.config.weight_format.value
        self.activation_bits = self.config.activation_format.value
        
        # Initialize scales as parameters so they can be learnt
        # We use one scale per entire tensor for both weights and activations
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('activation_scale', torch.ones(1))
        
        # For static quantization, we would need activation stats
        self.activation_stats = None
        if self.config.static_quantization:
            self.activation_stats = {
                'min': None,
                'max': None,
                'count': 0,
            }
    
    def _get_weight_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantization parameters for weights."""
        weight = self.linear.weight
        
        if self.config.weight_symmetric:
            # For symmetric quantization, we use the max absolute value
            weight_max = torch.max(torch.abs(weight)).detach()
            # If scale should be power of two, adjust it
            if self.config.power_of_two_scales:
                weight_max = torch.max(weight_max, torch.tensor(1e-8, device=weight_max.device))
                weight_scale = 2 ** torch.floor(torch.log2(weight_max))
            else:
                weight_scale = weight_max
            
            weight_scale = weight_scale / (2 ** (self.weight_bits - 1) - 1)
            weight_zero_point = torch.zeros_like(weight_scale)
        else:
            # For asymmetric quantization, we use min and max
            weight_min = torch.min(weight).detach()
            weight_max = torch.max(weight).detach()
            
            # Compute scale and zero point
            weight_scale = (weight_max - weight_min) / (2 ** self.weight_bits - 1)
            # Ensure scale is not too small
            weight_scale = torch.max(weight_scale, torch.tensor(1e-8, device=weight_scale.device))
            
            # If scale should be power of two, adjust it
            if self.config.power_of_two_scales:
                weight_scale = 2 ** torch.floor(torch.log2(weight_scale))
            
            weight_zero_point = torch.round(-weight_min / weight_scale)
        
        # Save the weight scale for later use
        self.weight_scale = weight_scale
        
        return weight_scale, weight_zero_point
    
    def _get_activation_qparams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantization parameters for activations."""
        if self.config.static_quantization and self.training:
            # In static quantization mode during training, collect stats
            if self.activation_stats['count'] == 0:
                self.activation_stats['min'] = torch.min(x).detach()
                self.activation_stats['max'] = torch.max(x).detach()
            else:
                self.activation_stats['min'] = torch.min(
                    self.activation_stats['min'], torch.min(x).detach()
                )
                self.activation_stats['max'] = torch.max(
                    self.activation_stats['max'], torch.max(x).detach()
                )
            self.activation_stats['count'] += 1
            
            # Use stored stats for quantization parameters
            act_min = self.activation_stats['min']
            act_max = self.activation_stats['max']
        else:
            # In dynamic quantization mode, use min/max of the current input
            act_min = torch.min(x).detach()
            act_max = torch.max(x).detach()
        
        if self.config.activation_symmetric:
            # For symmetric quantization, use max absolute value
            act_max = torch.max(torch.abs(act_min), torch.abs(act_max))
            
            # If scale should be power of two, adjust it
            if self.config.power_of_two_scales:
                act_max = torch.max(act_max, torch.tensor(1e-8, device=act_max.device))
                act_scale = 2 ** torch.floor(torch.log2(act_max))
            else:
                act_scale = act_max
            
            act_scale = act_scale / (2 ** (self.activation_bits - 1) - 1)
            act_zero_point = torch.zeros_like(act_scale)
        else:
            # For asymmetric quantization, use min and max
            act_scale = (act_max - act_min) / (2 ** self.activation_bits - 1)
            # Ensure scale is not too small
            act_scale = torch.max(act_scale, torch.tensor(1e-8, device=act_scale.device))
            
            # If scale should be power of two, adjust it
            if self.config.power_of_two_scales:
                act_scale = 2 ** torch.floor(torch.log2(act_scale))
            
            act_zero_point = torch.round(-act_min / act_scale)
        
        # Save the activation scale for later use
        self.activation_scale = act_scale
        
        return act_scale, act_zero_point
    
    def _fake_quantize(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, num_bits: int
    ) -> torch.Tensor:
        """
        Perform fake quantization on the input tensor.
        
        Args:
            x: Input tensor to quantize
            scale: Scale factor for quantization
            zero_point: Zero point for quantization
            num_bits: Number of bits to quantize to
            
        Returns:
            Fake quantized tensor (still in floating point)
        """
        # Compute quantization range
        qmin = 0 if zero_point.item() != 0 else -(2 ** (num_bits - 1))
        qmax = 2 ** num_bits - 1 if zero_point.item() != 0 else 2 ** (num_bits - 1) - 1
        
        # Scale and round
        x_scaled = x / scale + zero_point
        x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
        
        # Dequantize back to floating point
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization simulation."""
        # Get quantization parameters
        weight_scale, weight_zero_point = self._get_weight_qparams()
        act_scale, act_zero_point = self._get_activation_qparams(x)
        
        if self.training:
            # In training mode, fake quantize weights and activations
            weight_quant = self._fake_quantize(
                self.linear.weight, weight_scale, weight_zero_point, self.weight_bits
            )
            x_quant = self._fake_quantize(
                x, act_scale, act_zero_point, self.activation_bits
            )
            
            # Compute output with quantized weights and activations
            output = nn.functional.linear(
                x_quant, weight_quant, self.linear.bias
            )
            
            return output
        else:
            # In inference mode, we can use the original linear for now
            # Later we can implement actual quantized inference here
            return self.linear(x)
    
    def __repr__(self) -> str:
        return (
            f"QuantizedLinear(in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"weight_bits={self.weight_bits}, "
            f"activation_bits={self.activation_bits}, "
            f"weight_symmetric={self.config.weight_symmetric}, "
            f"activation_symmetric={self.config.activation_symmetric}, "
            f"static_quantization={self.config.static_quantization}, "
            f"power_of_two_scales={self.config.power_of_two_scales})"
        )


class QuantizationAwareTraining(ModelConverter):
    """
    Model converter for quantization-aware training.
    
    This converter replaces linear layers in the model with quantization-aware
    linear layers that simulate quantization during training.
    """
    
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        """
        Initialize quantization-aware training converter.
        
        Args:
            job_config: Job configuration
            parallel_dims: Parallel dimensions configuration
        """
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        
        # Parse quantization config from job config
        quant_config = getattr(job_config.experimental, "quantization", {})
        
        # Convert string enum values to actual enum values
        weight_format = getattr(quant_config, "weight_format", "INT8")
        activation_format = getattr(quant_config, "activation_format", "INT8")
        
        self.config = QuantizationConfig(
            weight_format=QuantFormat[weight_format],
            activation_format=QuantFormat[activation_format],
            weight_symmetric=getattr(quant_config, "weight_symmetric", True),
            activation_symmetric=getattr(quant_config, "activation_symmetric", True),
            static_quantization=getattr(quant_config, "static_quantization", False),
            power_of_two_scales=getattr(quant_config, "power_of_two_scales", True),
        )
        
        logger.info(f"Initialized quantization-aware training with config: {self.config}")
    
    def _convert_linear_module(self, module: nn.Module) -> None:
        """
        Recursively convert linear modules to quantized linear modules.
        
        Args:
            module: Module to convert
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace the linear module with a quantized linear module
                setattr(module, name, QuantizedLinear(child, self.config))
            else:
                # Recursively convert child modules
                self._convert_linear_module(child)
    
    def convert(self, model: nn.Module) -> None:
        """
        Convert the model to use quantization-aware training.
        
        Args:
            model: Model to convert
        """
        logger.info("Converting model to use quantization-aware training")
        self._convert_linear_module(model)
    
    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        """
        Post-optimizer hook for updating quantization parameters.
        
        This can be used for updating scales after each optimizer step,
        particularly useful for static quantization or when scales need
        to be synchronized across devices.
        
        Args:
            model: Model or list of model parts
        """
        # For now, no additional processing is needed after optimizer step
        pass


# Register the quantization-aware training converter
register_model_converter(QuantizationAwareTraining, "quantization_aware_training") 
