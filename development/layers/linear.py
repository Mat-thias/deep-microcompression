"""
@file linear.py
@brief PyTorch implementation of Linear layer with support for:
    1. Standard floating-point operation
    2. Dynamic/static quantization (per-tensor and per-channel)
    3. Channel pruning
    4. C code generation for deployment
"""

__all__ = [
    "Linear"
]

from typing import Optional, Tuple, Union
from functools import partial

import torch
from torch import nn

from .layer import Layer, Prune_Channel, Quantize

from ..utils import (
    convert_tensor_to_bytes_var,

    get_size_in_bits,

    STATIC_BIAS_BITWDHT,

    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity
)

class Linear(Layer, nn.Linear):
    """Quantization-aware Linear layer with support for:
        - Standard linear operation
        - Dynamic/static quantization (per-tensor and per-channel)
        - Model pruning
        - C code generation for deployment
    """

    def __init__(self, *args, **kwargs):
        """Initialize Linear layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or fake quantized)
            
        Returns:
            Output tensor according to current quantization mode
        """
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)

            if self.is_quantized:
                if hasattr(self, "input_quantize"):
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None:
                    bias = self.bias_quantize(bias)

        output = nn.functional.linear(input, weight, bias)
        
        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)

        return output
    

    @torch.no_grad()
    def init_prune_channel(
            self, 
            sparsity: Union[float, int], 
            keep_prev_channel_index: Optional[torch.Tensor], 
            input_shape: torch.Size,
            is_output_layer: bool = False, 
            metric: str = "l2"
        ) -> Optional[torch.Tensor]:
        """Prune channels based on importance metric
        
        Args:
            sparsity: Target sparsity ratio (0-1)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Importance metric ("l2" or absolute value)
            
        Returns:
            Indices of kept channels
        """
        super().init_prune_channel()

        if isinstance(sparsity, float):
            sparsity = min(max(0., sparsity), 1.)
            sparsity = int(sparsity * self.out_features)
        elif isinstance(sparsity, int): pass
        else:
            raise ValueError(f"sparsity must be of type int or float, got type {type(sparsity)}")

        sparsity = min(max(0, sparsity), self.out_features-1)
        density = self.out_features - sparsity


        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_features)

        if is_output_layer:
            # Skip pruning for output layer
            keep_current_channel_index = torch.arange(self.out_features)
        else:

            # Calculate channel importance
            importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
            channel_importance = importance.sum(dim=[1])
            keep_current_channel_index = torch.sort(torch.topk(channel_importance, density, dim=0).indices).values

        setattr(self, "weight_prune_channel", Prune_Channel(
            module=self, keep_current_channel_index=keep_current_channel_index, keep_prev_channel_index=keep_prev_channel_index
        ))

        if self.bias is not None:
            setattr(self, "bias_prune_channel", Prune_Channel(
                module=self, keep_current_channel_index=keep_current_channel_index
            ))

        return keep_current_channel_index


    @torch.no_grad()
    def init_quantize(self, bitwidth, scheme, granularity):
        if not self.is_pruned_channel:
            setattr(self, "weight_quantize", Quantize(
                self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC
            ))
        else:
            setattr(self, "weight_quantize", Quantize(
                self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, prune_channel=self.weight_prune_channel
            ))

        if scheme == QuantizationScheme.STATIC:
            setattr(self, "input_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
            ))
            setattr(self, "output_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
            ))

        if self.bias is not None:
            if not self.is_pruned_channel:
                if scheme == QuantizationScheme.DYNAMIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize]
                    ))
                elif scheme == QuantizationScheme.STATIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, STATIC_BIAS_BITWDHT, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize, self.input_quantize]
                    ))
            else:
                if scheme == QuantizationScheme.DYNAMIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize], prune_channel=self.bias_prune_channel
                    ))
                elif scheme == QuantizationScheme.STATIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, STATIC_BIAS_BITWDHT, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize, self.input_quantize], prune_channel=self.bias_prune_channel
                    ))

        # calibration
        if scheme == QuantizationScheme.DYNAMIC:
            self.weight_quantize.update_parameters(self.weight) 
            if self.bias is not None:
                self.bias_quantize.update_parameters(self.bias)
 

    @torch.no_grad()
    def get_size_in_bits(self):  
        weight, bias = self.get_compression_parameters()

        is_packed = False
        weight_bitwidth = None
        bias_bitwidth = None
        
        if self.is_quantized:
            is_packed = True
            weight_bitwidth = self.weight_quantize.bitwidth
            if self.bias is not None:
                bias_bitwidth = self.bias_quantize.bitwidth

        size = 0
        size += get_size_in_bits(weight, is_packed=is_packed, bitwidth=weight_bitwidth)
        if self.bias is not None:
            size += get_size_in_bits(bias, is_packed=is_packed, bitwidth=bias_bitwidth)  
        return size



    @torch.no_grad()
    def get_compression_parameters(self) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        weight = self.weight
        bias = self.bias

        if self.is_compressed:

            if self.is_pruned_channel:
                weight = self.weight_prune_channel.apply(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel.apply(bias)

            if self.is_quantized:
                    weight = self.weight_quantize.apply(weight)
                    if self.bias is not None:
                        bias = self.bias_quantize.apply(bias)
                    
        return weight, bias


    def get_output_tensor_shape(self, input_shape):
        out_features, _ = self.get_compression_parameters()[0].size()
        return torch.Size((out_features,)), torch.Size((out_features,))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """

        if self.bias is not None:
            weight, bias = self.get_compression_parameters()
            
            output_feature_size, input_feature_size = weight.size()

            weight_bitwidth = None
            if self.is_quantized:
                weight_bitwidth = self.weight_quantize.bitwidth
            # Convert weights to C representation
            param_header, param_def = convert_tensor_to_bytes_var(
                weight, 
                f"{var_name}_weight", 
                weight_bitwidth
            )   
            layer_header = param_header
            layer_param_def = param_def

            bias_bitwidth = None
            if self.is_quantized:
                bias_bitwidth = self.bias_quantize.bitwidth
            param_header, param_def = convert_tensor_to_bytes_var(
                bias, 
                f"{var_name}_bias",
                bias_bitwidth
            )
            layer_header += param_header
            layer_param_def += param_def

            q_type = None
            if self.is_quantized:
                q_type = self.weight_quantize.type

            if q_type is None or q_type == QUANTIZATION_NONE:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, (float*){var_name}_bias);\n"

            elif q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, *(float*){var_name}_weight_scale, (float*){var_name}_bias);\n"
                
                param_header, param_def = convert_tensor_to_bytes_var(
                                            self.weight_quantize.scale,
                                            f"{var_name}_weight_scale"
                                        )
                layer_header += param_header
                layer_param_def += param_def

            elif q_type == STATIC_QUANTIZATION_PER_TENSOR:

                layer_def = (
                    f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, "
                    f"{self.output_quantize.scale}, {self.output_quantize.zero_point}, {self.input_quantize.zero_point}, "
                    f"(int8_t*){var_name}_weight, (int32_t*){var_name}_bias, *(float*){var_name}_bias_scale);\n"
                ) 

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.output_quantize.scale, 
                    f"{var_name}_output_scale"
                )
                layer_header += param_header
                layer_param_def += param_def

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.output_quantize.zero_point, 
                    f"{var_name}_output_zero_point"
                )
                layer_header += param_header
                layer_param_def += param_def

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.bias_quantize.scale,
                    f"{var_name}_bias_scale"
                )
                layer_header += param_header
                layer_param_def += param_def
   
        
        else:
            weight = self.get_compression_parameters()
            
            output_feature_size, input_feature_size = weight.size()

            weight_bitwidth = None
            if self.is_quantized:
                weight_bitwidth = self.weight_quantize.bitwidth
            # Convert weights to C representation
            param_header, param_def = convert_tensor_to_bytes_var(
                weight, 
                f"{var_name}_weight", 
                weight_bitwidth
            )   
            layer_header = param_header
            layer_param_def = param_def


            q_type = None
            if self.is_quantized:
                q_type = self.weight_quantize.type


            if q_type is None or q_type == QUANTIZATION_NONE:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, nullptr);\n"

            elif q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, *(float*){var_name}_weight_scale, nullptr);\n"
                
                param_header, param_def = convert_tensor_to_bytes_var(
                                            self.weight_quantize.scale,
                                            f"{var_name}_weight_scale"
                                        )
                layer_header += param_header
                layer_param_def += param_def
            
            elif q_type == STATIC_QUANTIZATION_PER_TENSOR:

                layer_def = (
                    f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, "
                    f"{self.output_quantize.scale}, {self.output_quantize.zero_point}, {self.input_quantize.zero_point}, "
                    f"(int8_t*){var_name}_weight, nullptr, *(float*){var_name}_bias_scale);\n"
                ) 

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.output_quantize.scale, 
                    f"{var_name}_output_scale"
                )
                layer_header += param_header
                layer_param_def += param_def

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.output_quantize.zero_point, 
                    f"{var_name}_output_zero_point"
                )
                layer_header += param_header
                layer_param_def += param_def

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.bias_quantize.scale, 
                    f"{var_name}_bias_scale"
                )
                layer_header += param_header
                layer_param_def += param_def
   

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

        return layer_header, layer_def, layer_param_def