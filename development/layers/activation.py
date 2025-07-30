"""
@file activation.py
@brief PyTorch implementation of ReLU layer with support for:
    1. Standard floating-point operation
    2. Static quantization (per-tensor and per-channel)
    3. C code generation for deployment
"""

from typing import Optional

import torch
from torch import nn

from ..utils import (
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,

    quantize_per_tensor_assy,
    get_size_in_bits,

    convert_tensor_to_bytes_var
)
from .layer import Layer, Quantize


class ReLU(Layer, nn.ReLU):
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
        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "input_quantize"):
                    input = self.input_quantize(input)
        
        return super().forward(input)
    

    @torch.no_grad()
    def init_prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Optional[torch.Tensor], 
                     input_shape: torch.Size,
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
        # Nothing to do
        super().init_prune_channel()
        return keep_prev_channel_index

    def get_prune_channel_possible_hypermeters(self):
        return None
    

    def init_quantize(self, bitwidth, scheme, granularity):

        if scheme == QuantizationScheme.STATIC:
            setattr(self, "input_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
            ))


    def get_size_in_bits(self):
        if self.is_quantized:
            if hasattr(self, "input_quantize"):
                return get_size_in_bits(self.input_quantize.zero_point)
        return 0


    def get_compression_parameters(self):
        # Nothing to do 
        pass


    def get_output_tensor_shape(self, input_shape):
        # Nothing to do
        return input_shape, input_shape
    
    
    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generates C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_size = input_shape.numel()

        layer_param_def = ""
        layer_header = ""

        if self.is_quantized and hasattr(self, "input_quantize"):
            assert self.input_quantize.scheme == QuantizationScheme.STATIC, f"{self.__class__.__name__} has a input_quantize and is not static quantize"
            layer_def = f"{self.__class__.__name__} {var_name}({input_size}, *(float*){var_name}_input_zero_point);\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, 
                f"{var_name}_input_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def
        else:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def
    


class ReLU6(Layer, nn.ReLU6):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Clamped output tensor according to current quantization mode
        """

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "input_quantize"):
                    input = self.input_quantize(input)
        return super().forward(input)
    

    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        # Nothing to do
        super().init_prune_channel()
        return keep_prev_channel_index


    def get_prune_channel_possible_hypermeters(self):
        return None
    
    def init_quantize(self, bitwidth, scheme, granularity):
        if scheme == QuantizationScheme.STATIC:
            setattr(self, "input_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
            ))

    def get_size_in_bits(self):
        if self.is_quantized:
            if hasattr(self, "input_quantize"):
                return get_size_in_bits(self.input_quantize.zero_point)*2
        return 0

    def get_compression_parameters(self):
        # Nothing to do 
        pass


    def get_output_tensor_shape(self, input_shape):
        # Nothing to do
        return input_shape, input_shape
    
    
    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generates C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_size = input_shape.numel()


        layer_param_def = ""
        layer_header = ""

        if self.is_quantized and hasattr(self, "input_quantize"):
            assert self.input_quantize.scheme == QuantizationScheme.STATIC, f"{self.__class__.__name__} has a input_quantize and is not static quantize"
            layer_def = f"{self.__class__.__name__} {var_name}({input_size}, *(float*){var_name}_input_zero_point, *(float*){var_name}_input_six_point);\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, 
                f"{var_name}_input_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def

            input_six_point = quantize_per_tensor_assy(torch.Tensor([6]), self.input_quantize.scale, self.input_quantize.zero_point)
            param_header, param_def = convert_tensor_to_bytes_var(
                input_six_point, 
                f"{var_name}_input_six_point"
            )
            layer_header += param_header
            layer_param_def += param_def

        else:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def
    