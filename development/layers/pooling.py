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

from ..utils import (
    QUANTIZATION_NONE,
    DYNAMIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,
)


class MaxPool2d(Layer, nn.MaxPool2d):
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
    def prepare_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Union[torch.Tensor, None], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """Placeholder for channel pruning (MaxPool doesn't have weights to prune)
        
        Args:
            sparsity: Target sparsity ratio (unused)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer (unused)
            metric: Pruning metric (unused)
            
        Returns:
            Original channel indices (no pruning implemented)
        """
        super().prepare_prune_channel()
        # Nothing to do
        return keep_prev_channel_index


    def apply_prune_channel(self):
        super().apply_prune_channel()
        # Nothing to do
        pass

    
    def prepare_quantization(
        self, 
        bitwidth,
        type,
    ):
        super().prepare_quantization(bitwidth, type)
        # Nothing to do
        pass


    def apply_quantization(self):
        super().apply_quantization()
        # Nothing to do
        pass

    def prepare_dynamic_quantization_per_tensor(self, bitwidth):
        # Nothing to do
        pass

    def apply_dynamic_quantization_per_tensor(self):
        # Nothing to do
        pass


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


    def get_compression_parameters(self):
        #Nothing to do
        pass


    def get_output_tensor_shape(self, input_shape):
        
        C, H_in, W_in = input_shape
        
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        kH, kW = _pair(self.kernel_size)
        sH, sW = _pair(self.stride or self.kernel_size)  # PyTorch uses kernel_size as default if stride is None
        pH, pW = _pair(self.padding)
        
        # print(H_in, pH, kW, sW, self.kernel_size, self.stride, self.padding, self.dilation, isinstance(self.kernel_size, tuple))
        
        H_out = ((H_in + 2 * pH - kH) // sH) + 1
        W_out = ((W_in + 2 * pW - kW) // sW) + 1
        
        return torch.Size((C, H_out, W_out)), torch.Size((C, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_channel_size, input_row_size, input_col_size = input_shape
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
    



class AvgPool2d(Layer, nn.AvgPool2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass through max pooling layer
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Max pooled output tensor
        """
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
    def prepare_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Union[torch.Tensor, None], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """Placeholder for channel pruning (MaxPool doesn't have weights to prune)
        
        Args:
            sparsity: Target sparsity ratio (unused)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer (unused)
            metric: Pruning metric (unused)
            
        Returns:
            Original channel indices (no pruning implemented)
        """
        super().prepare_prune_channel()
        # Nothing to do
        return keep_prev_channel_index


    def apply_prune_channel(self):
        super().apply_prune_channel()
        # Nothing to do
        pass

    
    def prepare_quantization(
        self, 
        bitwidth,
        type,
    ):
        super().prepare_quantization(bitwidth, type)
        # Nothing to do
        pass


    def apply_quantization(self):
        super().apply_quantization()
        # Nothing to do
        pass

    def prepare_dynamic_quantization_per_tensor(self, bitwidth):
        # Nothing to do
        pass

    def apply_dynamic_quantization_per_tensor(self):
        # Nothing to do
        pass

    def get_compression_parameters(self):
        #Nothing to do
        pass


    def get_output_tensor_shape(self, input_shape):
        
        C, H_in, W_in = input_shape
        
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        kH, kW = _pair(self.kernel_size)
        sH, sW = _pair(self.stride or self.kernel_size)  # PyTorch uses kernel_size as default if stride is None
        pH, pW = _pair(self.padding)
        
        # print(H_in, pH, kW, sW, self.kernel_size, self.stride, self.padding, self.dilation, isinstance(self.kernel_size, tuple))
        
        H_out = ((H_in + 2 * pH - kH) // sH) + 1
        W_out = ((W_in + 2 * pW - kW) // sW) + 1

        return torch.Size((C, H_out, W_out)), torch.Size((C, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_channel_size, input_row_size, input_col_size = input_shape
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