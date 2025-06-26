"""
@file activation.py
@brief PyTorch implementation of ReLU layer with support for:
    1. Standard floating-point operation
    2. Static quantization (per-tensor and per-channel)
    3. C code generation for deployment
"""

from typing import Union
import torch
from torch import nn
from ..utilis import (
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,
    QUANTIZATION_NONE,

    get_size_in_bits
)

class ReLU(nn.ReLU):
    """Quantization-aware ReLU layer with support for:
        - Standard ReLU operation
        - Quantized inference modes
        - Model pruning (placeholder)
        - C code generation for deployment
    """

    def __init__(self, *args, **kwargs):
        """Initialize ReLU layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Clamped output tensor according to current quantization mode
        """
        # Store input shape for later use in code generation
        setattr(self, "input_shape", input.size())
        
        # Determine minimum value based on quantization mode
        min_val = 0  # Default for non-quantized case
        
        if hasattr(self, "quantization_type"):
            if getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
                min_val = self.input_zero_point
            elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
                min_val = self.input_zero_point

        return torch.clamp(input, min=min_val)
    

    def get_size_in_bits(self):
        if hasattr(self, "input_zero_point"):
            return get_size_in_bits(self.input_zero_point)
        return 0


    @torch.no_grad()
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Union[torch.Tensor, None], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        """Placeholder for channel pruning functionality
        
        Args:
            sparsity: Target sparsity ratio (unused)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Pruning metric (unused)
            
        Returns:
            Original channel indices (no pruning implemented)
        """
        return keep_prev_channel_index

    @torch.no_grad()
    def static_quantize_per_tensor(self,
                                 input_batch_real: torch.Tensor,
                                 input_batch_quant: torch.Tensor,
                                 input_scale: torch.Tensor,
                                 input_zero_point: torch.Tensor,
                                 bitwidth: int = 8):
        """Configures per-tensor static quantization
        
        Args:
            input_batch_real: FP32 input samples
            input_batch_quant: Quantized input samples
            input_scale: Quantization scale factor
            input_zero_point: Quantization zero point
            bitwidth: Quantization bitwidth (unused)
            
        Returns:
            Tuple of (real_output, quant_output, scale, zero_point)
        """
        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_TENSOR)
        self.register_buffer("input_zero_point", input_zero_point)

        # Apply ReLU in both real and quantized domains
        output_batch_real = torch.clamp(input_batch_real, min=0)
        output_batch_quant = torch.clamp(input_batch_quant, min=input_zero_point)

        return output_batch_real, output_batch_quant, input_scale, input_zero_point

    @torch.no_grad()
    def static_quantize_per_channel(self,
                                  input_batch_real: torch.Tensor,
                                  input_batch_quant: torch.Tensor,
                                  input_scale: torch.Tensor,
                                  input_zero_point: torch.Tensor,
                                  bitwidth: int = 8):
        """Configures per-channel static quantization
        
        Args:
            Same as static_quantize_per_tensor
            
        Returns:
            Same as static_quantize_per_tensor
        """
        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)
        self.register_buffer("input_zero_point", input_zero_point)

        # Apply ReLU in both real and quantized domains
        output_batch_real = torch.clamp(input_batch_real, min=0)
        output_batch_quant = torch.clamp(input_batch_quant, min=input_zero_point)

        return output_batch_real, output_batch_quant, input_scale, input_zero_point

    @torch.no_grad()
    def convert_to_c(self, var_name):
        """Generates C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_size = self.input_shape[1:].numel()

        if getattr(self, "quantization_type", QUANTIZATION_NONE) != STATIC_QUANTIZATION_PER_TENSOR:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
        else:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size}, {self.input_zero_point});\n"

        layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
        layer_param_def = ""
        
        return layer_header, layer_def, layer_param_def