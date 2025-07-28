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

class Flatten(Layer, nn.Flatten):
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
        
        # return input.flatten(self.start_dim, self.end_dim)
        return super().forward(input)
    


    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Union[torch.Tensor, None], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """Coordinate channel pruning between layers by adjusting channel indices
        
        Args:
            sparsity: Target sparsity ratio (unused, maintained for interface consistency)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Pruning metric (unused, maintained for interface consistency)
            
        Returns:
            Adjusted channel indices accounting for flatten operation
        """
        super().init_prune_channel()

        # Calculate number of elements per channel in original input
        channel_numel = input_shape[1:].numel()

        if is_output_layer:
            pass
            # Output layer doesn't prune, just pass through

        # Calculate start positions for each kept channel
        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(input_shape[0])
        start_positions = keep_prev_channel_index * channel_numel
        channel_elements_index = torch.arange(channel_numel).to(start_positions.device)

        # Generate indices for all elements in kept channels
        keep_current_channel_index = start_positions.view(-1, 1) + channel_elements_index

        return keep_current_channel_index.flatten()
    
    
    def init_quantize(self, bitwidth, scheme, granularity):
        # Nothing to do
        pass


    def get_size_in_bits(self):
        return 0

    def get_compression_parameters(self):
        # Nothing to do
        pass

    def get_output_tensor_shape(self, input_shape):

        return torch.Size((input_shape.numel(),)), torch.Size((input_shape.numel(),))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_size = input_shape.numel()
        

        layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
        layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
        layer_param_def = ""

        return layer_header, layer_def, layer_param_def