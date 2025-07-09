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

class BatchNorm2d(Layer, nn.BatchNorm2d):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def forward(self, input):
        input = super().forward(input)
        return input
   
    
    @torch.no_grad
    def prepare_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        super().prepare_prune_channel()
        self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"] = keep_prev_channel_index
        
        return keep_prev_channel_index
    
    @torch.no_grad
    def apply_prune_channel(self):
        super().apply_prune_channel()
        
        mask = torch.zeros_like(self.weight)
        keep_prev_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"]
        mask[keep_prev_channel_index] = 1

        self.weight.mul_(mask)
        self.bias.mul_(mask)
        if self.running_mean is not None and self.running_var is not None:
            self.running_mean.mul_(mask)
            
        return
    
    def apply_prune_channel_external(self, weight, bias, running_mean=None, running_var=None):
        
        keep_prev_channel_index = self.__dict__["_dmc"]["keep_prev_channel_index"]

        weight = torch.index_select(self.weight, 0, keep_prev_channel_index)
        bias = torch.index_select(self.bias, 0, keep_prev_channel_index)

        if running_mean is not None and running_var is not None:
            running_mean = torch.index_select(running_mean, 0, keep_prev_channel_index)
            running_var = torch.index_select(running_var, 0, keep_prev_channel_index)
            return weight, bias, running_mean, running_var
        return weight, bias


    def prepare_quantization(
        self, 
        bitwidth,
        type,               
        input_batch_real = None, 
        input_batch_quant = None,
        input_scale = None, 
        input_zero_point = None,
    ):
        super().prepare_quantization(bitwidth, type)
        
        if type == DYNAMIC_QUANTIZATION_PER_TENSOR:
            return self.prepare_dynamic_quantization_per_tensor(bitwidth)

        elif type == STATIC_QUANTIZATION_PER_TENSOR: 
            if input_batch_real is None or input_batch_quant is None or input_scale is None or input_zero_point is None:
                raise ValueError("Pass the calibration parameters for static quantization!")
            return self.prepare_static_quantization_per_tensor(    
                bitwidth,        
                input_batch_real, 
                input_batch_quant,
                input_scale, 
                input_zero_point,
            )



    def apply_quantization(self):
        super().apply_quantization()
        pass

    def prepare_dynamic_quantization_per_tensor(self, bitwidth):
        pass

    def apply_dynamic_quantization_per_tensor(self):
        pass


    def prepare_static_quantization_per_tensor(
        self,                     
        bitwidth,
        input_batch_real, 
        input_batch_quant,
        input_scale, 
        input_zero_point,
    ):
        self.__dict__["_dmc"]["quantization"]["input_zero_point"] = input_zero_point

        raise NotImplementedError("Static Quantization not implemented for batchnorm")
        return output_batch_real, output_batch_quant, input_scale, input_zero_point
        


    def apply_static_quantization_per_tensor(self):
        pass


    def get_size_in_bits(self):
        
        weight, bias, running_mean, running_var = self.get_compression_parameters()
        size = 0

        size += get_size_in_bits(weight)
        size += get_size_in_bits(bias)
        size += get_size_in_bits(running_mean)
        size += get_size_in_bits(running_var)

        return size

    def get_compression_parameters(self):

        weight = self.weight
        bias = self.bias
        running_mean = self.running_mean
        running_var = self.running_var

        if "prune_channel" in self.__dict__["_dmc"]:
            weight, bias, running_mean, running_var = self.apply_prune_channel_external(weight, bias, running_mean, running_var)

        return weight, bias, running_mean, running_var

    def get_output_tensor_shape(self, input_shape):
        # Nothing to do
        return input_shape, input_shape

    
    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):

        weight, bias, running_mean, running_var = self.get_compression_parameters(return_param=True)

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
    





































     

    # @torch.no_grad()
    # def set_compression_parameters(self):

    #     if getattr(self, "pruned", False):
    #         self.set_prune_parameters()

    #     return
    

    # @torch.no_grad()
    # def get_compression_parameters(self, return_param=False):
        
    #     weight = self.weight
    #     bias = self.bias
    #     running_mean = self.running_mean
    #     running_var = self.running_var

    #     if return_param:
    #         if getattr(self, "pruned", False):
    #             weight, bias, running_mean, running_var = self.get_prune_parameters(return_param=True)

    #         return weight, bias, running_mean, running_var
        
    #     return