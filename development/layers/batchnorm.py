from typing import Optional

import torch
from torch import nn

from .layer import Layer

from ..utils import (
    get_quantize_scale_per_channel_sy,
    get_quantize_scale_per_tensor_sy,
    get_quantize_scale_zero_point_per_tensor_assy,
    dequantize_per_tensor_sy, 
    dequantize_per_channel_sy,
    quantize_per_tensor_assy,
    quantize_per_channel_sy,
    quantize_per_tensor_sy,
    convert_tensor_to_bytes_var,
    QUANTIZATION_NONE,
    DYNAMIC_QUANTIZATION_PER_TENSOR,
    DYNAMIC_QUANTIZATION_PER_CHANNEL,
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,

    get_size_in_bits
)

class BatchNorm2d(nn.BatchNorm2d, Layer):


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    
    def forward(self, input):

        input = super().forward(input)
        return input
    

    @torch.no_grad()
    def set_compression_parameters(self):

        if getattr(self, "pruned", False):
            self.set_prune_parameters()

        return
    

    @torch.no_grad()
    def get_compression_parameters(self):
        
        weight = self.weight
        bias = self.bias
        running_mean = self.running_mean
        running_var = self.running_var

        if getattr(self, "pruned", False):
            weight, bias, running_mean, running_var = self.get_prune_parameters()

        return weight, bias, running_mean, running_var
    
    
    @torch.no_grad
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Optional[torch.Tensor], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        super().prune_channel(sparsity, keep_prev_channel_index, is_output_layer, metric)

        setattr(self, "keep_prev_channel_index", keep_prev_channel_index)
        
        self.set_prune_parameters()

        return keep_prev_channel_index
    
    @torch.no_grad
    def set_prune_parameters(self):
        
        mask = torch.zeros_like(self.weight)
        mask[getattr(self, "keep_prev_channel_index")] = 1

        self.weight.mul_(mask)
        self.bias.mul_(mask)
        self.running_mean.mul_(mask)
        self.running_var.mul_(mask)

        return 
    
    @torch.no_grad
    def get_prune_parameters(self):
        
        weight = torch.index_select(self.weight, 0, getattr(self, "keep_prev_channel_index"))
        bias = torch.index_select(self.bias, 0, getattr(self, "keep_prev_channel_index"))
        running_mean = torch.index_select(self.running_mean, 0, getattr(self, "keep_prev_channel_index"))
        running_var = torch.index_select(self.running_var, 0, getattr(self, "keep_prev_channel_index"))

        return weight, bias, running_mean, running_var
        

    def get_size_in_bits(self):
        
        weight, bias, running_mean, running_var = self.get_compression_parameters()
        size = 0

        size += get_size_in_bits(weight)
        size += get_size_in_bits(bias)
        size += get_size_in_bits(running_mean)
        size += get_size_in_bits(running_var)

        return size


    def get_output_tensor_shape(self, input_shape):

        return input_shape

    
    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):

        weight, bias, running_mean, running_var = self.get_compression_parameters()

        input_row_size, input_col_size = input_shape[1:]

        input_channel_size = weight.size(0)

        param_header, param_def = convert_tensor_to_bytes_var(weight, f"{var_name}_weight")
        layer_header = param_header
        layer_param_def = param_def

        param_header, param_def = convert_tensor_to_bytes_var(bias, f"{var_name}_bias")
        layer_header += param_header
        layer_param_def += param_def

        param_header, param_def = convert_tensor_to_bytes_var(running_mean, f"{var_name}_running_mean")
        layer_header += param_header
        layer_param_def += param_def

        param_header, param_def = convert_tensor_to_bytes_var(torch.sqrt(running_var+self.eps), f"{var_name}_running_var")
        layer_header += param_header
        layer_param_def += param_def


        if getattr(self, "quantization_type", QUANTIZATION_NONE) != STATIC_QUANTIZATION_PER_TENSOR:
            layer_def = (
                f"{self.__class__.__name__} {var_name}({input_channel_size}, {input_row_size}, {input_col_size}, "
                f"(float*){var_name}_weight, (float*){var_name}_bias, (float*){var_name}_running_mean, (float*){var_name}_running_var);\n"
            )
        # else:
        #     layer_def = f"{self.__class__.__name__} {var_name}({input_size}, {self.input_zero_point});\n"

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def
    