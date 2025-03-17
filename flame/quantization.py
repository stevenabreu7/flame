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
import torch.nn.functional as F
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


class FakeQuantize(nn.Module):
    """
    Base class for fake quantization modules.
    
    Implements common functionality for fake quantization that simulates
    quantization during training, while keeping the values in floating point.
    """
    
    def __init__(
        self,
        config: QuantizationConfig,
        is_weight: bool = True,
    ):
        super().__init__()
        self.config = config
        self.is_weight = is_weight
        
        # Set up bit width and symmetric flag based on whether this is for weights or activations
        if is_weight:
            self.num_bits = self.config.weight_format.value
            self.symmetric = self.config.weight_symmetric
        else:
            self.num_bits = self.config.activation_format.value
            self.symmetric = self.config.activation_symmetric
        
        # Initialize scale as a buffer (not a parameter)
        self.register_buffer('scale', torch.ones(1))
        
        # For static quantization of activations, we need to collect statistics
        self.stats = None
        if not is_weight and self.config.static_quantization:
            self.stats = {
                'min': None,
                'max': None,
                'count': 0,
            }
    
    def _get_qparams(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization parameters (scale and zero_point).
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of scale and zero_point tensors
        """
        if self.symmetric:
            # For symmetric quantization, we use the max absolute value
            if self.is_weight or not self.config.static_quantization or not self.training:
                # For weights or dynamic quantization, use current tensor
                x_abs_max = torch.max(torch.abs(x)).detach()
            else:
                # For static quantization during training, update stats
                if self.stats['count'] == 0:
                    self.stats['min'] = torch.min(x).detach()
                    self.stats['max'] = torch.max(x).detach()
                else:
                    self.stats['min'] = torch.min(
                        self.stats['min'], torch.min(x).detach()
                    )
                    self.stats['max'] = torch.max(
                        self.stats['max'], torch.max(x).detach()
                    )
                self.stats['count'] += 1
                
                # Use stats for scale calculation
                x_abs_max = torch.max(
                    torch.abs(self.stats['min']),
                    torch.abs(self.stats['max'])
                )
            
            # Ensure x_abs_max is not too small
            x_abs_max = torch.max(x_abs_max, torch.tensor(1e-8, device=x_abs_max.device))
            
            # Adjust scale to be a power of two if required
            if self.config.power_of_two_scales:
                scale = 2 ** torch.floor(torch.log2(x_abs_max))
            else:
                scale = x_abs_max
            
            # Compute final scale
            scale = scale / (2 ** (self.num_bits - 1) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            # For asymmetric quantization
            if self.is_weight or not self.config.static_quantization or not self.training:
                # For weights or dynamic quantization, use current tensor
                x_min = torch.min(x).detach()
                x_max = torch.max(x).detach()
            else:
                # For static quantization during training, update and use stats
                if self.stats['count'] == 0:
                    self.stats['min'] = torch.min(x).detach()
                    self.stats['max'] = torch.max(x).detach()
                else:
                    self.stats['min'] = torch.min(
                        self.stats['min'], torch.min(x).detach()
                    )
                    self.stats['max'] = torch.max(
                        self.stats['max'], torch.max(x).detach()
                    )
                self.stats['count'] += 1
                
                x_min = self.stats['min']
                x_max = self.stats['max']
            
            # Compute scale and zero point
            scale = (x_max - x_min) / (2 ** self.num_bits - 1)
            scale = torch.max(scale, torch.tensor(1e-8, device=scale.device))
            
            # Adjust scale to be a power of two if required
            if self.config.power_of_two_scales:
                scale = 2 ** torch.floor(torch.log2(scale))
            
            zero_point = torch.round(-x_min / scale)
        
        # Save the scale for later use
        self.scale = scale
        
        return scale, zero_point
    
    def fake_quantize(
        self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply fake quantization to a tensor.
        
        Args:
            x: Input tensor
            scale: Scale factor
            zero_point: Zero point offset
            
        Returns:
            Fake quantized tensor (still in floating point)
        """
        # Compute quantization range
        qmin = 0 if zero_point.item() != 0 else -(2 ** (self.num_bits - 1))
        qmax = 2 ** self.num_bits - 1 if zero_point.item() != 0 else 2 ** (self.num_bits - 1) - 1
        
        # Scale and round to simulate quantization
        x_scaled = x / scale + zero_point
        x_quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
        
        # Dequantize back to floating point
        x_dequant = (x_quant - zero_point) * scale
        
        return x_dequant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization in the forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Fake quantized tensor
        """
        if self.training or not self.is_weight or self.config.static_quantization:
            # Get quantization parameters and apply fake quantization
            scale, zero_point = self._get_qparams(x)
            return self.fake_quantize(x, scale, zero_point)
        else:
            # During inference for non-static weights, we can skip quantization
            return x


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
        
        # Create fake quantizers for weights and activations
        self.weight_quantizer = FakeQuantize(config, is_weight=True)
        self.activation_quantizer = FakeQuantize(config, is_weight=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization simulation."""
        # Quantize input activations
        x_quant = self.activation_quantizer(x)
        
        # Quantize weights
        weight_quant = self.weight_quantizer(self.linear.weight)
        
        # Compute output with quantized weights and activations
        output = nn.functional.linear(
            x_quant, weight_quant, self.linear.bias
        )
        
        return output
    
    def __repr__(self) -> str:
        return (
            f"QuantizedLinear(in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"weight_bits={self.weight_quantizer.num_bits}, "
            f"activation_bits={self.activation_quantizer.num_bits}, "
            f"weight_symmetric={self.config.weight_symmetric}, "
            f"activation_symmetric={self.config.activation_symmetric}, "
            f"static_quantization={self.config.static_quantization}, "
            f"power_of_two_scales={self.config.power_of_two_scales})"
        )


class QuantizedRMSNorm(nn.Module):
    """
    Quantization-aware RMSNorm layer.
    
    This layer applies fake quantization to the inputs and outputs
    of RMSNorm during training.
    """
    
    def __init__(
        self,
        norm_module: Union[nn.RMSNorm, nn.Module],
        config: QuantizationConfig,
    ):
        super().__init__()
        self.norm = norm_module
        self.config = config
        
        # Create fake quantizers for activations
        self.input_quantizer = FakeQuantize(config, is_weight=False)
        self.output_quantizer = FakeQuantize(config, is_weight=False)
        
        # Create fake quantizer for weight (if it exists)
        if hasattr(norm_module, 'weight') and norm_module.weight is not None:
            self.weight_quantizer = FakeQuantize(config, is_weight=True)
        else:
            self.weight_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization simulation."""
        # Quantize input activations
        x_quant = self.input_quantizer(x)
        
        # Apply normalization (with quantized weights if available)
        if self.weight_quantizer is not None and hasattr(self.norm, 'weight'):
            # Store original weight
            orig_weight = self.norm.weight.data
            
            # Temporarily replace with quantized weight
            weight_quant = self.weight_quantizer(self.norm.weight)
            self.norm.weight.data = weight_quant
            
            # Forward pass
            output = self.norm(x_quant)
            
            # Restore original weight
            self.norm.weight.data = orig_weight
        else:
            # Forward pass without weight quantization
            output = self.norm(x_quant)
        
        # Quantize output activations
        output_quant = self.output_quantizer(output)
        
        return output_quant
    
    def __repr__(self) -> str:
        return (
            f"QuantizedRMSNorm(normalized_shape={self.norm.normalized_shape if hasattr(self.norm, 'normalized_shape') else 'unknown'}, "
            f"weight_bits={self.weight_quantizer.num_bits if self.weight_quantizer else None}, "
            f"activation_bits={self.input_quantizer.num_bits})"
        )


class QuantizedGatedMLP(nn.Module):
    """
    Quantization-aware GatedMLP (SwiGLU) module.
    
    This module quantizes the internal linear layers and activations of the MLP.
    """
    
    def __init__(
        self,
        mlp_module: nn.Module,  # Assuming HGRNMLP or equivalent
        config: QuantizationConfig,
    ):
        super().__init__()
        self.mlp = mlp_module
        self.config = config
        
        # Create fake quantizers for activations at different points in the forward pass
        self.input_quantizer = FakeQuantize(config, is_weight=False)
        self.gate_quantizer = FakeQuantize(config, is_weight=False)
        self.up_quantizer = FakeQuantize(config, is_weight=False)
        self.pre_down_quantizer = FakeQuantize(config, is_weight=False)
        self.output_quantizer = FakeQuantize(config, is_weight=False)
        
        # The linear layers are quantized separately by QuantizedLinear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization simulation."""
        # We assume that the linear layers are already quantized,
        # so we only need to quantize the activations between them
        
        # Quantize input
        x_quant = self.input_quantizer(x)
        
        # Get gate and up projections
        gate = self.mlp.gate_proj(x_quant)
        y = self.mlp.up_proj(x_quant)
        
        # Quantize intermediate activations
        gate_quant = self.gate_quantizer(gate)
        y_quant = self.up_quantizer(y)
        
        if self.mlp.fuse_swiglu and hasattr(self.mlp, 'swiglu_linear'):
            # Handle fused swiglu case
            out = self.mlp.swiglu_linear(
                gate_quant, y_quant, 
                self.mlp.down_proj.weight, 
                self.mlp.down_proj.bias
            )
        else:
            # Handle normal case
            swiglu_out = F.silu(gate_quant) * y_quant
            swiglu_out_quant = self.pre_down_quantizer(swiglu_out)
            out = self.mlp.down_proj(swiglu_out_quant)
        
        # Quantize output
        out_quant = self.output_quantizer(out)
        
        return out_quant
    
    def __repr__(self) -> str:
        return (
            f"QuantizedGatedMLP(hidden_size={self.mlp.hidden_size}, "
            f"intermediate_size={self.mlp.intermediate_size}, "
            f"activation_bits={self.input_quantizer.num_bits})"
        )


class QuantizedHGRNAttention(nn.Module):
    """
    Quantization-aware HGRNAttention module.
    
    This module quantizes the internal linear layers and activations
    of the HGRN Attention mechanism.
    """
    
    def __init__(
        self,
        attn_module: nn.Module,  # HGRNAttention
        config: QuantizationConfig,
    ):
        super().__init__()
        self.attn = attn_module
        self.config = config
        
        # Create fake quantizers for activations
        self.input_quantizer = FakeQuantize(config, is_weight=False)
        self.i_quantizer = FakeQuantize(config, is_weight=False)
        self.f_quantizer = FakeQuantize(config, is_weight=False)
        self.g_quantizer = FakeQuantize(config, is_weight=False)
        self.output_quantizer = FakeQuantize(config, is_weight=False)
        
        # The linear layers and g_norm will be quantized separately
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ):
        """Forward pass with quantization simulation."""
        # Quantize input
        x_quant = self.input_quantizer(hidden_states)
        
        # Apply projections
        i = self.attn.i_proj(x_quant)
        f = self.attn.f_proj(x_quant)
        g = self.attn.g_proj(x_quant)
        
        # Quantize intermediate activations
        i_quant = self.i_quantizer(i)
        f_quant = self.f_quantizer(f)
        g_quant = self.g_quantizer(g)
        
        # Apply optional convolutions
        if self.attn.use_short_conv:
            if hasattr(self.attn, 'i_conv1d'):
                i_quant = self.attn.i_conv1d(i_quant)
            if hasattr(self.attn, 'f_conv1d'):
                f_quant = self.attn.f_conv1d(f_quant)
            if hasattr(self.attn, 'q_conv1d'):
                g_quant = self.attn.q_conv1d(g_quant)
        
        # Apply normalization and gate
        g_out = self.attn.g_norm(g_quant)
        
        # Main attention logic depends on mode
        last_state = past_key_values[self.attn.layer_idx] if past_key_values is not None and len(past_key_values) > self.attn.layer_idx else None
        
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        if self.attn.mode == 'chunk':
            # Chunk-wise HGRN
            from fla.ops.hgrn import chunk_hgrn
            h = chunk_hgrn(i_quant, f_quant, attention_mask, last_state)
        else:
            # Fused recurrent HGRN
            from fla.ops.hgrn import fused_recurrent_hgrn
            h = fused_recurrent_hgrn(i_quant, f_quant, attention_mask, last_state)
        
        # Apply output projection
        out = self.attn.o_proj(h * g_out)
        
        # Quantize output
        out_quant = self.output_quantizer(out)
        
        if use_cache:
            # Return output and updated cache
            if self.attn.mode == 'chunk':
                # Handle state appropriately for chunk mode
                state = h[:, -1:].detach().contiguous()
            else:
                # Handle state appropriately for fused recurrent mode
                state = h[:, -1:].detach().contiguous()
            
            return out_quant, None, state
        
        return out_quant, None, None
    
    def __repr__(self) -> str:
        return (
            f"QuantizedHGRNAttention(hidden_size={self.attn.hidden_size}, "
            f"mode={self.attn.mode}, "
            f"activation_bits={self.input_quantizer.num_bits})"
        )


class QuantizedFusedRMSNormSwishGate(nn.Module):
    """
    Quantization-aware FusedRMSNormSwishGate module.
    
    This module quantizes the inputs and outputs of the fused
    RMSNorm with Swish Gate module.
    """
    
    def __init__(
        self,
        norm_module: nn.Module,  # FusedRMSNormSwishGate
        config: QuantizationConfig,
    ):
        super().__init__()
        self.norm = norm_module
        self.config = config
        
        # Create fake quantizers for activations
        self.input_quantizer = FakeQuantize(config, is_weight=False)
        self.output_quantizer = FakeQuantize(config, is_weight=False)
        
        # Create fake quantizer for weights if they exist
        if hasattr(norm_module, 'weight') and norm_module.weight is not None:
            self.weight_quantizer = FakeQuantize(config, is_weight=True)
        else:
            self.weight_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization simulation."""
        # Quantize input
        x_quant = self.input_quantizer(x)
        
        # Apply normalization and gate (with quantized weights if applicable)
        if self.weight_quantizer is not None and hasattr(self.norm, 'weight'):
            # Store original weight
            orig_weight = self.norm.weight.data
            
            # Temporarily replace with quantized weight
            weight_quant = self.weight_quantizer(self.norm.weight)
            self.norm.weight.data = weight_quant
            
            # Forward pass
            output = self.norm(x_quant)
            
            # Restore original weight
            self.norm.weight.data = orig_weight
        else:
            # Forward without weight quantization
            output = self.norm(x_quant)
        
        # Quantize output
        output_quant = self.output_quantizer(output)
        
        return output_quant


class QuantizationAwareTraining(ModelConverter):
    """
    Model converter for quantization-aware training.
    
    This converter replaces modules in the model with quantization-aware versions
    that simulate quantization during training.
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
    
    def _convert_module(self, module: nn.Module) -> None:
        """
        Recursively convert modules to their quantized versions.
        
        Args:
            module: Module to convert
        """
        # First check for modules to replace directly
        for name, child in list(module.named_children()):
            # Check module type and replace with appropriate quantized version
            if isinstance(child, nn.Linear):
                logger.info(f"Quantizing Linear module: {name}")
                setattr(module, name, QuantizedLinear(child, self.config))
            elif type(child).__name__ == 'RMSNorm' or isinstance(child, nn.RMSNorm):
                logger.info(f"Quantizing RMSNorm module: {name}")
                setattr(module, name, QuantizedRMSNorm(child, self.config))
            elif type(child).__name__ == 'FusedRMSNormSwishGate':
                logger.info(f"Quantizing FusedRMSNormSwishGate module: {name}")
                setattr(module, name, QuantizedFusedRMSNormSwishGate(child, self.config))
            elif type(child).__name__ == 'HGRNMLP' or type(child).__name__ == 'GatedMLP':
                logger.info(f"Quantizing GatedMLP module: {name}")
                setattr(module, name, QuantizedGatedMLP(child, self.config))
            elif type(child).__name__ == 'HGRNAttention':
                logger.info(f"Quantizing HGRNAttention module: {name}")
                setattr(module, name, QuantizedHGRNAttention(child, self.config))
            else:
                # Recursively process children
                self._convert_module(child)
    
    def convert(self, model: nn.Module) -> None:
        """
        Convert the model to use quantization-aware training.
        
        Args:
            model: Model to convert
        """
        logger.info("Converting model to use quantization-aware training")
        self._convert_module(model)
    
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
