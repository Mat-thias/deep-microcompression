"""
@file flatten.py
@brief PyTorch implementation of Flatten layer with support for:
    1. Standard flatten operation
    2. Static quantization pass-through
    3. Channel pruning coordination
    4. C code generation for deployment
"""

from typing import Union
import torch
from torch import nn

from .layer import Layer

from ..utilis import (
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,
)

class Flatten(nn.Flatten, Layer):
    """Quantization-aware Flatten layer that maintains:
        - Standard flatten functionality
        - Quantization state pass-through
        - Channel pruning coordination
        - C code generation support
    
    Note: This layer doesn't perform actual quantization but preserves quantization
    parameters through the network for adjacent layers.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Flatten layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass that flattens input while preserving shape information
        
        Args:
            input: Input tensor of any shape
            
        Returns:
            Flattened tensor according to start_dim and end_dim
        """
        # Store input shape for later use in pruning and code generation
        setattr(self, "input_shape", input.size())
        # return input.flatten(self.start_dim, self.end_dim)
        return super().forward(input)
    


    def get_size_in_bits(self):
        return 0

    @torch.no_grad()
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Union[torch.Tensor, None], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        """Coordinate channel pruning between layers by adjusting channel indices
        
        Args:
            sparsity: Target sparsity ratio (unused, maintained for interface consistency)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Pruning metric (unused, maintained for interface consistency)
            
        Returns:
            Adjusted channel indices accounting for flatten operation
        """
        setattr(self, "pruned", True)

        # Calculate number of elements per channel in original input
        channel_numel = self.input_shape[2:].numel()

        # if is_output_layer:
        #     # Output layer doesn't prune, just pass through
        #     keep_current_channel_index = None
        #     return keep_current_channel_index

        # Calculate start positions for each kept channel
        start_positions = keep_prev_channel_index * channel_numel
        channel_elements_index = torch.arange(channel_numel)

        # Generate indices for all elements in kept channels
        keep_current_channel_index = start_positions.view(-1, 1) + channel_elements_index

        return keep_current_channel_index.flatten()

    @torch.no_grad()
    def static_quantize_per_tensor(self,
                                 input_batch_real: torch.Tensor,
                                 input_batch_quant: torch.Tensor,
                                 input_scale: torch.Tensor,
                                 input_zero_point: torch.Tensor,
                                 bitwidth: int = 8):
        """Pass-through static per-tensor quantization parameters
        
        Args:
            input_batch_real: FP32 input samples
            input_batch_quant: Quantized input samples
            input_scale: Input quantization scale
            input_zero_point: Input quantization zero point
            bitwidth: Quantization bitwidth (unused, maintained for interface)
            
        Returns:
            Tuple of (real_output, quant_output, scale, zero_point)
        """
        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_TENSOR)

        # Simply flatten both real and quantized inputs
        output_batch_real = input_batch_real.flatten(self.start_dim, self.end_dim)
        output_batch_quant = input_batch_quant.flatten(self.start_dim, self.end_dim)

        return output_batch_real, output_batch_quant, input_scale, input_zero_point

    @torch.no_grad()
    def static_quantize_per_channel(self,
                                  input_batch_real: torch.Tensor,
                                  input_batch_quant: torch.Tensor,
                                  input_scale: torch.Tensor,
                                  input_zero_point: torch.Tensor,
                                  bitwidth: int = 8):
        """Pass-through static per-channel quantization parameters
        
        Args:
            Same as static_quantize_per_tensor
            
        Returns:
            Same as static_quantize_per_tensor
        """
        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)

        # Simply flatten both real and quantized inputs
        output_batch_real = input_batch_real.flatten(self.start_dim, self.end_dim)
        output_batch_quant = input_batch_quant.flatten(self.start_dim, self.end_dim)

        return output_batch_real, output_batch_quant, input_scale, input_zero_point

    def get_output_tensor_shape(self, input_shape):

        return torch.Size((input_shape.numel(),))
    

    @torch.no_grad()
    def convert_to_c(self, var_name):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_size = self.input_shape[1:].numel()
        

        layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
        layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
        layer_param_def = ""

        return layer_header, layer_def, layer_param_def