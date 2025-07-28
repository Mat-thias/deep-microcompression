"""
@file conv.py
@brief PyTorch implementation of Conv2d layer with support for:
    1. Standard floating-point operation
    2. Dynamic/static quantization (per-tensor and per-channel)
    3. Channel pruning
    4. C code generation for deployment
"""

__all__ = ["Conv2d"]

from typing import Optional, Tuple, Union
from functools import partial

import torch
from torch import nn

from .layer import Layer, Prune_Channel, Quantize

from ..utils import (
    convert_tensor_to_bytes_var,

    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,

    STATIC_BIAS_BITWDHT,

    get_size_in_bits
)


class Conv2d(Layer, nn.Conv2d):
# class Conv2d(nn.Conv2d, Layer):
    """Quantization-aware Conv2d layer with support for:
        - Standard convolution operation
        - Dynamic/static quantization (per-tensor and per-channel)
        - Channel pruning
        - C code generation for deployment
    """

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d layer with standard PyTorch parameters"""
        self.pad = kwargs.pop("pad", (0, 0, 0, 0))
        assert len(self.pad) == 4, "Invalid pad, pad = (pad_left, pad_right, pad_top, pad_bottom)"
        
        groups = kwargs.pop("groups", 1)
        assert groups == 1 or groups == self.out_channels, "Only supports group = 1 or group = out_channels (Depthwise conv)"

        if "padding" in kwargs:
            assert kwargs["padding"] == 0, "Use pad instead of padding to pad input"
        super().__init__(*args, **kwargs)


    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Output tensor after convolution with quantization if enabled
        """
        # Perform convolution with appropriate padding
        
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

        input =  nn.functional.pad(input, self.pad, "constant", 0) 
        output = nn.functional.conv2d(
            input, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)

        return output


    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """Prune channels based on importance metric
        
        Args:
            sparsity: Target sparsity ratio (0-1)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Importance metric ("l2" or other)
            
        Returns:
            Indices of kept channels
        """
        assert(self.groups == 1 or self.groups == self.out_channels), "Channel Pruning is yet to be implement for grouped convolution, on Deepwise convolution."
        super().init_prune_channel()

        if isinstance(sparsity, float):
            sparsity = min(max(0., sparsity), 1.)
            sparsity = int(sparsity * self.out_channels)
        elif isinstance(sparsity, int): pass
        else:
            raise ValueError(f"sparsity must be of type int or float, got type {type(sparsity)}")

        sparsity = min(max(0, sparsity), self.out_channels-1)
        density = self.out_channels - sparsity

        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_channels)

        if self.groups == self.out_channels:

            keep_prev_channel_index_temp = keep_prev_channel_index
            keep_prev_channel_index = torch.arange(1)

            if is_output_layer:
                keep_current_channel_index = torch.arange(self.out_channels)
            else:
                keep_current_channel_index = keep_prev_channel_index_temp
        else:

            if is_output_layer:
                keep_current_channel_index = torch.arange(self.out_channels)

            else:
                importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
                channel_importance = importance.sum(dim=[1, 2, 3])
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
    def get_size_in_bits(self) -> int:
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

        C_in, H_in, W_in = input_shape
        
        # Unpack parameters (handle both int and tuple)
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        # kH, kW = _pair(self.kernel_size)
        C_out, _, kH, kW = self.get_compression_parameters()[0].size()
            
        sH, sW = _pair(self.stride)
        dH, dW = _pair(self.dilation)

        pW = self.pad[0] + self.pad[1]
        pH = self.pad[2] + self.pad[3]
        
        H_out = ((H_in +  pH - dH * (kH - 1) - 1) // sH) + 1
        W_out = ((W_in +  pW - dW * (kW - 1) - 1) // sW) + 1
        
        return torch.Size((C_in, H_in +  pH, W_in +  pW)), torch.Size((C_out, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        if self.bias is not None:
            weight, bias = self.get_compression_parameinput_shapeters()

            input_channel_size, input_row_size, input_col_size = input_shape

            output_channel_size, _,\
            kernel_row_size, kernel_col_size = weight.size()
            stride_row, stride_col = self.stride
            pad = self.pad

            if self.groups == self.out_channels:
                groups = input_channel_size
            else:
                groups = self.groups

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
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(float*){var_name}_weight, (float*){var_name}_bias);\n"
                )
            elif q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(int8_t*){var_name}_weight,  *(float*){var_name}_weight_scale, (float*){var_name}_bias);\n"
                )                
                param_header, param_def = convert_tensor_to_bytes_var(
                                            self.weight_quantize.scale,
                                            f"{var_name}_weight_scale"
                                        )
                layer_header += param_header
                layer_param_def += param_def

            elif q_type == STATIC_QUANTIZATION_PER_TENSOR:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"{self.output_quantize.scale}, {self.output_quantize.zero_point}, "
                    f"{self.input_quantize.zero_point}, (int8_t*){var_name}_weight, "
                    f"(int32_t*){var_name}_bias, *(float*){var_name}_bias_scale);\n"
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

            input_channel_size, input_row_size, input_col_size = input_shape

            output_channel_size, _,\
            kernel_row_size, kernel_col_size = weight.size()
            stride_row, stride_col = self.stride
            pad = self.pad

            if self.groups == self.in_channels:
                groups = input_channel_size
            else:
                groups = self.groups

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
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(float*){var_name}_weight, nullptr);\n"
                )
            elif q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(int8_t*){var_name}_weight,  *(float*){var_name}_weight_scale, nullptr);\n"
                )                
                param_header, param_def = convert_tensor_to_bytes_var(
                                            self.weight_quantize.scale,
                                            f"{var_name}_weight_scale"
                                        )
                layer_header += param_header
                layer_param_def += param_def

            elif q_type == STATIC_QUANTIZATION_PER_TENSOR:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"{self.output_quantize.scale}, {self.output_quantize.zero_point}, "
                    f"{self.input_quantize.zero_point}, (int8_t*){var_name}_weight, "
                    f"nullptr, *(float*){var_name}_bias_scale);\n"
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

