"""
@file pooling.py
@brief PyTorch implementation of MaxPool2d layer with support for:
    1. Standard max pooling operation
    2. Static quantization (per-tensor and per-channel)
    3. C code generation for deployment
"""

from typing import Union

import torch
from torch import nn

from .layer import Layer

from ..utilis import (
    QUANTIZATION_NONE,
    DYNAMIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,
)


class MaxPool2d(nn.MaxPool2d, Layer):
    """Quantization-aware MaxPool2d layer with support for:
        - Standard max pooling operation
        - Quantized inference modes
        - C code generation for deployment
    """

    def __init__(self, *args, **kwargs):
        """Initialize MaxPool2d layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass through max pooling layer
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Max pooled output tensor
        """
        # Store input shape for later use in code generation
        setattr(self, "input_shape", input.size())

        # return nn.functional.max_pool2d(
        #     input,
        #     self.kernel_size,
        #     self.stride,
        #     self.padding,
        #     self.dilation,
        #     ceil_mode=self.ceil_mode,
        #     return_indices=self.return_indices,
        # )
        return super().forward(input)

    def get_size_in_bits(self):
        return 0

    @torch.no_grad()
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Union[torch.Tensor, None], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        """Placeholder for channel pruning (MaxPool doesn't have weights to prune)
        
        Args:
            sparsity: Target sparsity ratio (unused)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer (unused)
            metric: Pruning metric (unused)
            
        Returns:
            Original channel indices (no pruning implemented)
        """
        setattr(self, "pruned", True)

        return keep_prev_channel_index

    @torch.no_grad()
    def static_quantize_per_tensor(self,
                                 input_batch_real: torch.Tensor,
                                 input_batch_quant: torch.Tensor,
                                 input_scale: torch.Tensor,
                                 input_zero_point: torch.Tensor,
                                 bitwidth: int = 8):
        """Configure static per-tensor quantization for MaxPool
        
        Args:
            input_batch_real: FP32 input samples
            input_batch_quant: Quantized input samples
            input_scale: Input quantization scale
            input_zero_point: Input quantization zero point
            bitwidth: Quantization bitwidth (unused)
            
        Returns:
            Tuple of (real_output, quant_output, input_scale, input_zero_point)
        """
        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_TENSOR)

        # Apply MaxPool in real (float) domain
        output_batch_real = torch.nn.functional.max_pool2d(
            input_batch_real,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # Apply MaxPool in quantized domain - same parameters
        output_batch_quant = torch.nn.functional.max_pool2d(
            input_batch_quant,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        return output_batch_real, output_batch_quant, input_scale, input_zero_point

    @torch.no_grad()
    def static_quantize_per_channel(self,
                                  input_batch_real: torch.Tensor,
                                  input_batch_quant: torch.Tensor,
                                  input_scale: torch.Tensor,
                                  input_zero_point: torch.Tensor,
                                  bitwidth: int = 8):
        """Configure static per-channel quantization for MaxPool
        
        Args:
            Same as static_quantize_per_tensor
            
        Returns:
            Same as static_quantize_per_tensor
        """
        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)

        # Apply MaxPool in real (float) domain
        output_batch_real = torch.nn.functional.max_pool2d(
            input_batch_real,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        # Apply MaxPool in quantized domain - same parameters
        output_batch_quant = torch.nn.functional.max_pool2d(
            input_batch_quant,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

        return output_batch_real, output_batch_quant, input_scale, input_zero_point
    
    def get_output_tensor_shape(self, input_shape):
        
        C, H_in, W_in = input_shape
        
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        kH, kW = _pair(self.kernel_size)
        sH, sW = _pair(self.stride or self.kernel_size)  # PyTorch uses kernel_size as default if stride is None
        pH, pW = _pair(self.padding)
        dH, dW = _pair(self.dilation)
        
        H_out = ((H_in + 2 * pH - dH * (kH - 1) - 1) // sH) + 1
        W_out = ((W_in + 2 * pW - dW * (kW - 1) - 1) // sW) + 1
        
        return torch.Size((C, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_channel_size, input_row_size, input_col_size = self.input_shape[1:]
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        layer_def = (
            f"{self.__class__.__name__} {var_name}("
            f"{input_channel_size}, {input_row_size}, {input_col_size}, "
            f"{kernel_size}, {stride}, {padding});\n"
        )
        layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
        layer_param_def = ""

        return layer_header, layer_def, layer_param_def