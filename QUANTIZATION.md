# HGRN Model Quantization

This document describes the quantization-aware training (QAT) implementation for the HGRN (Hierarchically Gated Recurrent Neural Network) model architecture.

## HGRN Model Architecture

The HGRN model is a transformer-like architecture designed for sequence modeling, with some key differences from standard transformer models. The architecture consists of:

1. **Token Embeddings**: Standard embedding layer to convert token IDs to vectors
2. **HGRN Blocks**: The core building blocks of the architecture, each containing:
   - **Attention Normalization**: RMSNorm layer (or Dynamic Tanh Norm in DyT version)
   - **HGRN Attention**: A specialized attention mechanism that differs from standard self-attention
   - **MLP Normalization**: Another RMSNorm layer (or Dynamic Tanh Norm in DyT version)
   - **MLP**: Gated MLP that uses SwiGLU activation
3. **Final LayerNorm**: Normalization before output projection
4. **Output Projection**: Linear layer to project to vocabulary dimension

### HGRN Attention

The HGRN Attention differs significantly from standard transformer attention:

1. **Input Projections**: `i_proj`, `f_proj`, and `g_proj` (all Linear layers)
2. **Optional Convolutions**: Short 1D convolutions on the projections
3. **FusedRMSNormSwishGate**: Applied to the gate projection
4. **Hierarchical Gated Recurrent Mechanism**: Either chunk-based or fully recurrent
5. **Output Projection**: `o_proj` (Linear layer)

### Gated MLP

The HGRN MLP uses a SwiGLU activation and contains:

1. **Gate Projection**: `gate_proj` (Linear layer) 
2. **Value Projection**: `up_proj` (Linear layer)
3. **SwiGLU Activation**: Either fused or unfused implementation
4. **Output Projection**: `down_proj` (Linear layer)

## Quantization Implementation

Our quantization implementation supports quantization-aware training of the HGRN model, with different formats for weights and activations.

### Quantized Components

The following components are quantized:

| Component | Module Type | What is Quantized | Notes |
|-----------|-------------|-------------------|-------|
| Linear Layers | `nn.Linear` | Weights and activations | Used throughout the model for all projections |
| RMSNorm | `nn.RMSNorm` or custom `RMSNorm` | Weights (if present), input and output activations | Used for normalization layers |
| Dynamic Tanh Norm | Custom DyT modules | Weights (if present), input and output activations | Used in DyT version of the model |
| FusedRMSNormSwishGate | Custom module | Weights (if present), input and output activations | Used in HGRN Attention |
| Gated MLP | Custom `HGRNMLP` or `GatedMLP` | All intermediate activations | Used as the feed-forward network |
| HGRN Attention | Custom `HGRNAttention` | All intermediate activations | The core attention mechanism |

### Quantization Features

The implementation supports:

1. **Dynamic Quantization**: Compute quantization parameters on-the-fly during inference
2. **Static Quantization**: Pre-compute quantization parameters during calibration
3. **Different Formats**: INT4, INT8, FLOAT8 (and extensible to others)
4. **Symmetric or Asymmetric**: Support for both quantization modes
5. **Power-of-Two Scales**: Option to constrain scales to powers of two for hardware efficiency

### Notable Implementation Details

1. **Per-Tensor Scales**: Quantization scales are shared across the entire tensor
2. **Power-of-Two Scales**: For hardware efficiency, scales can be constrained to powers of two
3. **Triton Kernels**: Native Triton kernels (like in fused operations) are not quantized directly

## Quantization Flow

1. **Configuration**: Set quantization parameters in the training script or TOML config
2. **Module Conversion**: During model initialization, standard modules are replaced with quantized versions
3. **Training**: Quantization-aware training simulates quantization effects while keeping weights in floating point
4. **Inference**: Either dynamic quantization (on-the-fly) or static quantization (pre-calibrated)

## Limitations

1. **Triton Kernels**: Native Triton kernels are not directly quantized
2. **Performance**: The current implementation focuses on accuracy over performance optimization
3. **Hardware-Specific Optimizations**: Not yet implemented

## Future Work

1. **Static Quantization**: Enhance support for calibration-based static quantization
2. **Triton Kernel Integration**: Extend quantization to specialized Triton kernels
3. **Hardware Acceleration**: Implement optimized kernels for accelerated quantized inference
4. **Additional Formats**: Support more quantization formats and mixed precision 
