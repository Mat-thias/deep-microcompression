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
        
        input = super().forward(
            nn.functional.pad(input, self.pad, "constant", 0) 
        )

        return input
    

    def get_size_in_bits(self) -> int:
        
        size = 0
        if self.bias is not None:
            weight, bias = self.get_compression_parameters()
        else:
            weight = self.get_compression_parameters()

        size += get_size_in_bits(weight)
        if self.bias is not None:
            size += get_size_in_bits(bias)

        return size



    @torch.no_grad()
    def prepare_prune_channel(
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
        assert(self.groups == 1 or self.groups == self.in_channels), "Channel Pruning is yet to be implement for grouped convolution, on Deepwise convolution."
        super().prepare_prune_channel()

        sparsity = min(max(0., sparsity), 1.)


        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_channels)

        if self.groups == 1:
            prune_groups = 1
            # setattr(self, "prune_groups", 1)
            
            if is_output_layer:
                keep_current_channel_index = torch.arange(self.out_channels)

            else:
                importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
                importance = importance.sum(dim=[1, 2, 3])
                threshold = importance.quantile(sparsity)
                keep_current_channel_index = torch.nonzero(
                    (importance >= threshold).to(torch.int32)
                ).squeeze(dim=1)
            setattr(self, "keep_current_channel_index", keep_current_channel_index)

        else:# self.groups == self.in_channels:
            prune_groups = keep_prev_channel_index.size(0)
            # setattr(self, "prune_groups", keep_prev_channel_index.size(0))

            keep_prev_channel_index_temp = keep_prev_channel_index
            
            keep_prev_channel_index = torch.arange(1)

            if is_output_layer:
                keep_current_channel_index = torch.arange(self.out_channels)
 
            else:
                keep_current_channel_index = keep_prev_channel_index_temp
        
        self.__dict__["_dmc"]["prune_channel"]["groups"] = prune_groups
        self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"] = keep_prev_channel_index
        self.__dict__["_dmc"]["prune_channel"]["keep_current_channel_index"] = keep_current_channel_index

        return keep_current_channel_index
    

    @torch.no_grad()
    def apply_prune_channel(self) -> None:
        super().apply_prune_channel()
        
                                 
        weight_mask_prev_channel = torch.zeros_like(self.weight)
        weight_mask_current_channel = torch.zeros_like(self.weight)

        keep_prev_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"]
        keep_current_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_current_channel_index"]

        weight_mask_prev_channel[:,keep_prev_channel_index,:,:] = 1
        weight_mask_current_channel[keep_current_channel_index,:,:,:] = 1
        weight_mask = torch.mul(weight_mask_prev_channel, weight_mask_current_channel)
        
        self.weight.mul_(weight_mask)

        if self.bias is not None:
            bias_mask = torch.zeros_like(self.bias)
            bias_mask[keep_current_channel_index] = 1
            
            self.bias.mul_(bias_mask)
        
        return
    
    def apply_prune_channel_external(self, weight, bias=None):

        if "prune_channel" not in self.__dict__["_dmc"]:
            raise AttributeError("Layer must be prepared before applying compression")
        
        keep_prev_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"]
        keep_current_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_current_channel_index"]

        weight = torch.index_select(weight, 1, keep_prev_channel_index)
        weight = torch.index_select(weight, 0, keep_current_channel_index)

        if self.bias is not None:
            bias = torch.index_select(self.bias, 0, keep_current_channel_index)
            return weight, bias
        
        return weight


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
            self.prepare_dynamic_quantization_per_tensor(bitwidth)

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

        q_type = self.__dict__["_dmc"]["quantization"]["type"]  
        if q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
            return self.apply_dynamic_quantization_per_tensor() 


    def apply_quantization_external(self, weight, bias=None):
          
        if "quantization" not in self.__dict__["_dmc"]:
            raise AttributeError("Layer must be prepared before applying compression")
        
        q_type = self.__dict__["_dmc"]["quantization"]["type"]                     

        if q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
            return self.apply_dynamic_quantization_per_tensor_external(weight, bias) 
        elif q_type == STATIC_QUANTIZATION_PER_TENSOR:
            return self.apply_static_quantization_per_tensor_external(weight, bias)


    def prepare_dynamic_quantization_per_tensor(self, bitwidth):

        weight_scale = get_quantize_scale_per_tensor_sy(self.weight, bitwidth)
        self.__dict__["_dmc"]["quantization"]["weight_scale"] = weight_scale


    @torch.no_grad()
    def apply_dynamic_quantization_per_tensor(self):
        weight_scale = self.__dict__["_dmc"]["quantization"]["weight_scale"]
        bitwidth = self.__dict__["_dmc"]["quantization"]["bitwidth"]

        weight_quant = quantize_per_tensor_sy(self.weight, weight_scale, bitwidth)
        self.weight.copy_(dequantize_per_tensor_sy(weight_quant, weight_scale))

        return
    
    @torch.no_grad()
    def apply_dynamic_quantization_per_tensor_external(self, weight, bias=None):
        weight_scale = self.__dict__["_dmc"]["quantization"]["weight_scale"]
        bitwidth = self.__dict__["_dmc"]["quantization"]["bitwidth"]

        weight_quant = quantize_per_tensor_sy(weight, weight_scale, bitwidth)

        return weight_quant
    

    def prepare_static_quantization_per_tensor(
        self,                     
        bitwidth,
        input_batch_real, 
        input_batch_quant,
        input_scale, 
        input_zero_point,
    ):
        weight_scale = get_quantize_scale_per_tensor_sy(self.weight, bitwidth)
        weight_quant = quantize_per_tensor_sy(self.weight, weight_scale, bitwidth)
        
        if self.bias is not None:
            bias_quant = quantize_per_tensor_sy(
                self.bias, 
                (weight_scale * input_scale), 
                bitwidth * 4, 
                torch.int32
            ).to(torch.int32)
        else:
            bias_quant = None

        # Perform quantized convolution, output in int32
        output_batch_quant_bias_dtype = torch.nn.functional.conv2d(
            (input_batch_quant.to(torch.int32) - input_zero_point),
            weight_quant.to(torch.int32),
            bias_quant
        )

        output_batch_real = torch.nn.functional.conv2d(
            input_batch_real,
            self.weight,
            self.bias
        )

        # Calculate output quantization parameters
        output_scale, output_zero_point = get_quantize_scale_zero_point_per_tensor_assy(
            output_batch_real, 
            bitwidth
        )

        #converting from int32 to int8
        output_batch_quant = quantize_per_tensor_assy(
            output_batch_quant_bias_dtype * input_scale * weight_scale,
            output_scale,
            output_zero_point,
            bitwidth
        )

        self.__dict__["_dmc"]["quantization"]["weight_scale"] = weight_scale
        self.__dict__["_dmc"]["quantization"]["input_zero_point"] = input_zero_point
        self.__dict__["_dmc"]["quantization"]["bias_scale"] = (weight_scale * input_scale)
        self.__dict__["_dmc"]["quantization"]["output_scale"] = output_scale
        self.__dict__["_dmc"]["quantization"]["output_zero_point"] = output_zero_point

        return output_batch_real, output_batch_quant, output_scale, output_zero_point


    @torch.no_grad()
    def apply_static_quantization_per_tensor(self):
        weight_scale = self.__dict__["_dmc"]["quantization"]["weight_scale"]
        bitwidth = self.__dict__["_dmc"]["quantization"]["bitwidth"]

        weight_quant = quantize_per_tensor_sy(self.weight, weight_scale, bitwidth)
        self.weight.copy_(dequantize_per_tensor_sy(weight_quant, weight_scale))

        return
    
    @torch.no_grad()
    def apply_static_quantization_per_tensor_external(self, weight, bias=None):
        weight_scale = self.__dict__["_dmc"]["quantization"]["weight_scale"]
        bitwidth = self.__dict__["_dmc"]["quantization"]["bitwidth"]

        weight_quant = quantize_per_tensor_sy(weight, weight_scale, bitwidth)

        return weight_quant




    @torch.no_grad()
    def get_compression_parameters(self) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
        if self.bias is not None:
            weight = self.weight
            bias = self.bias

            if "prune_channel" in self.__dict__["_dmc"]:
                weight, bias = self.apply_prune_channel_external(weight, bias)
            if "quantization" in self.__dict__["_dmc"]:
                weight, bias = self.apply_quantization_external(weight, bias)

            return weight, bias
    
        weight = self.weight

        if getattr(self, "pruned", False):
            weight = self.apply_prune_channel_external(weight)
        if "quantization" in self.__dict__["_dmc"]:
            weight = self.apply_quantization_external(weight)

        return weight



    def get_output_tensor_shape(self, input_shape):

        C_in, H_in, W_in = input_shape
        
        # Unpack parameters (handle both int and tuple)
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        # kH, kW = _pair(self.kernel_size)
        if self.bias is not None:
            C_out, _, kH, kW = self.get_compression_parameters()[0].size()
        else:
            C_out, _, kH, kW = self.get_compression_parameters().size()
            
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
            weight, bias = self.get_compression_parameters()
            input_channel_size, input_row_size, input_col_size = input_shape

            output_channel_size, _,\
            kernel_row_size, kernel_col_size = weight.size()
            stride_row, stride_col = self.stride
            pad = self.pad
            groups = getattr(self, "prune_groups", self.groups)

            # Convert weights and parameters to C byte arrays
            param_header, param_def = convert_tensor_to_bytes_var(
                weight, 
                f"{var_name}_weight", 
                getattr(self, "quantization_bitwidth", 8)
            )
            layer_header = param_header
            layer_param_def = param_def

            param_header, param_def = convert_tensor_to_bytes_var(bias, f"{var_name}_bias")
            layer_header += param_header
            layer_param_def += param_def

            # Generate layer definition based on quantization mode
            if not hasattr(self, "quantization_type") or getattr(self, "quantization_type") == QUANTIZATION_NONE:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(float*){var_name}_weight, (float*){var_name}_bias);\n"
                )

        else:
            weight = self.get_compression_parameters()
            input_channel_size, input_row_size, input_col_size = input_shape

            output_channel_size, _,\
            kernel_row_size, kernel_col_size = weight.size()
            stride_row, stride_col = self.stride
            pad = self.pad
            groups = getattr(self, "prune_groups", self.groups)

            # Convert weights and parameters to C byte arrays
            param_header, param_def = convert_tensor_to_bytes_var(
                weight, 
                f"{var_name}_weight", 
                getattr(self, "quantization_bitwidth", 8)
            )
            layer_header = param_header
            layer_param_def = param_def

            # Generate layer definition based on quantization mode
            if not hasattr(self, "quantization_type") or getattr(self, "quantization_type") == QUANTIZATION_NONE:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(float*){var_name}_weight, nullptr);\n"
                )
    
        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

        return layer_header, layer_def, layer_param_def











































    # def forward(self, input:torch.Tensor) -> torch.Tensor:
    #     """Forward pass with quantization awareness
        
    #     Args:
    #         input: Input tensor (float or quantized)
            
    #     Returns:
    #         Output tensor after convolution with quantization if enabled
    #     """
    #     # Use pruned or quantized weights if available
    #     # weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
    #     # bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias
        
    #     # if getattr(self, "pruned", False):
    #     #     self.set_()
    #         # print("conv prunned")

    #     # # Handle different quantization modes
    #     # if hasattr(self, "quantization_type"):
    #     #     if getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
    #     #         weight = dequantize_per_tensor_sy(self.weight_dmc, self.weight_scale)
    #     #     elif getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_CHANNEL:
    #     #         weight = dequantize_per_channel_sy(self.weight_dmc, self.weight_scale)
    #     #     elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
    #     #         input = input.to(torch.int32) - self.input_zero_point
    #     #         weight = self.weight_dmc.to(torch.int32)
    #     #         bias = self.bias_dmc
    #     #     elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
    #     #         input = input.to(torch.int32) - self.input_zero_point
    #     #         weight = self.weight_dmc.to(torch.int32)
    #     #         bias = self.bias_dmc

    #     # Perform convolution with appropriate padding
    #     if self.pad:
    #         input = nn.functional.pad(input, self.pad, "constant", 0) 
    #     input = super().forward(input)
    #     # if self.padding_mode != "zeros":
    #     #     input = nn.functional.conv2d(
    #     #         nn.functional.pad(
    #     #             input, self._reversed_padding_repeated_twice, mode=self.padding_mode
    #     #         ),
    #     #         weight,
    #     #         bias,
    #     #         self.stride,
    #     #         (0, 0),
    #     #         self.dilation,
    #     #         self.groups,
    #     #     )
    #     # else:
    #     #     input = nn.functional.conv2d(
    #     #         input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    #     #     )

    #     # Apply output quantization if in static quantization mode
    #     # if hasattr(self, "quantization_type"):
    #     #     if getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
    #     #         input = quantize_per_tensor_assy(
    #     #             input * self.bias_scale, 
    #     #             self.output_scale, 
    #     #             self.output_zero_point, 
    #     #             self.quantization_bitwidth
    #     #         )
    #     #     elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
    #     #         input = quantize_per_tensor_assy(
    #     #             input * self.bias_scale.view(1, -1, 1, 1), 
    #     #             self.output_scale, 
    #     #             self.output_zero_point, 
    #     #             self.quantization_bitwidth
    #     #         )

    #     return input
    

    # def get_size_in_bits(self) -> int:
        
    #     # # Use pruned or quantized weights if available
    #     # weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
    #     # bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias
    #     size = 0

    #     if self.bias is not None:
    #         weight, bias = self.get_compression_parameters()
    #     else:
    #         weight = self.get_compression_parameters()

    #     if getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
    #         size += get_size_in_bits(weight)
    #         if self.bias is not None:
    #             size += get_size_in_bits(bias)
    #     # else:
    #     #     size += get_size_in_bits(weight, is_packed=True, bitwidth=self.quantization_bitwidth)
    #     #     size += get_size_in_bits(bias, is_packed=True, bitwidth=self.quantization_bitwidth)

    #     # # Handle different quantization modes
    #     # if hasattr(self, "quantization_type"):
    #     #     if getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR or getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_CHANNEL:
    #     #         size += get_size_in_bits(self.weight_scale)
    #     #     elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR or getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
    #     #         size += get_size_in_bits(self.input_zero_point)
    #     #         size += get_size_in_bits(self.bias_scale)
    #     #         size += get_size_in_bits(self.output_scale)
    #             # size += get_size_in_bits(self.output_zero_point)

    #     return size


    # @torch.no_grad()
    # def set_compression_parameters(self) -> None:

    #     if getattr(self, "pruned", False):
    #         self.set_prune_parameters()

    #     return
    

    # @torch.no_grad()
    # def get_compression_parameters(self) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
    #     if self.bias is not None:
    #         weight = self.weight
    #         bias = self.bias

    #         if getattr(self, "pruned", False):
    #             weight, bias = self.apply_prune_channel(return_param=True)
    #         return weight, bias
    
    #     weight = self.weight

    #     if getattr(self, "pruned", False):
    #         weight = self.apply_prune_channel(return_param=True)

    #     return weight


    # @torch.no_grad()
    # def prepare_prune_channel(self, 
    #                  sparsity: float, 
    #                  keep_prev_channel_index: Optional[torch.Tensor], 
    #                  input_shape: torch.Size,
    #                  is_output_layer: bool = False, 
    #                  metric: str = "l2"):
    #     """Prune channels based on importance metric
        
    #     Args:
    #         sparsity: Target sparsity ratio (0-1)
    #         keep_prev_channel_index: Channels to keep from previous layer
    #         is_output_layer: Flag if this is an output layer
    #         metric: Importance metric ("l2" or other)
            
    #     Returns:
    #         Indices of kept channels
    #     """
    #     setattr(self, "pruned", True)

    #     assert(self.groups == 1 or self.groups == self.in_channels), "Channel Pruning is yet to be implement for grouped convolution, on Deepwise convolution."

    #     sparsity = min(max(0., sparsity), 1.)


    #     if keep_prev_channel_index is None:
    #         keep_prev_channel_index = torch.arange(self.in_channels)

    #     # weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
    #     # bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias
    #     if self.groups == 1:

    #         setattr(self, "prune_groups", 1)
            
    #         if is_output_layer:
    #             keep_current_channel_index = torch.arange(self.out_channels)

    #         else:
    #             importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
    #             importance = importance.sum(dim=[1, 2, 3])
    #             threshold = importance.quantile(sparsity)
    #             keep_current_channel_index = torch.nonzero(
    #                 (importance >= threshold).to(torch.int32)
    #             ).squeeze(dim=1)
    #         setattr(self, "keep_current_channel_index", keep_current_channel_index)

    #     else:# self.groups == self.in_channels:
    #         setattr(self, "prune_groups", keep_prev_channel_index.size(0))

    #         keep_prev_channel_index_temp = keep_prev_channel_index
            
    #         keep_prev_channel_index = torch.arange(1)

    #         if is_output_layer:
    #             keep_current_channel_index = torch.arange(self.out_channels)
 
    #         else:
    #             keep_current_channel_index = keep_prev_channel_index_temp

    #     setattr(self, "keep_prev_channel_index", keep_prev_channel_index)
    #     setattr(self, "keep_current_channel_index", keep_current_channel_index)
 
    #     return keep_current_channel_index
    

    # @torch.no_grad()
    # def apply_prune_channel(self) -> Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        
    #     weight_mask_prev_channel = torch.zeros_like(self.weight)
    #     weight_mask_current_channel = torch.zeros_like(self.weight)

    #     weight_mask_prev_channel[:,getattr(self, "keep_prev_channel_index"),:,:] = 1
    #     weight_mask_current_channel[getattr(self, "keep_current_channel_index"),:,:,:] = 1
    #     weight_mask = torch.mul(weight_mask_prev_channel, weight_mask_current_channel)
        
    #     self.weight.mul_(weight_mask)

    #     if self.bias is not None:
    #         bias_mask = torch.zeros_like(self.bias)
    #         bias_mask[getattr(self, "keep_current_channel_index")] = 1
            
    #         self.bias.mul_(bias_mask)
        
    #     if return_param:
    #         weight = torch.index_select(self.weight, 1, getattr(self, "keep_prev_channel_index"))
    #         weight = torch.index_select(weight, 0, getattr(self, "keep_current_channel_index"))

    #         if self.bias is not None:
    #             bias = torch.index_select(self.bias, 0, getattr(self, "keep_current_channel_index"))
    #             return weight, bias
            
    #         return weight
    #     return
        


    # @torch.no_grad()
    # def get_prune_parameters(self) -> :

    #     weight = torch.index_select(self.weight, 1, getattr(self, "keep_prev_channel_index"))
    #     weight = torch.index_select(weight, 0, getattr(self, "keep_current_channel_index"))

    #     if self.bias is not None:
    #         bias = torch.index_select(self.bias, 0, getattr(self, "keep_current_channel_index"))
    #         return weight, bias
    #     return weight

    # @torch.no_grad()
    # def apply_prune_channel(self, inplace:bool=True) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:

    #     weight_mask_prev_channel = torch.zeros_like(self.weight)
    #     weight_mask_current_channel = torch.zeros_like(self.weight)
    #     bias_mask = torch.zeros_like(self.bias)

    #     try:
    #         if inplace:
    #             weight_mask_prev_channel[:,getattr(self, "keep_prev_channel_index"),:,:] = 1
    #             weight_mask_current_channel[getattr(self, "keep_current_channel_index"),:,:,:] = 1
    #             weight_mask = torch.mul(weight_mask_prev_channel, weight_mask_current_channel)
    #             # print(weight_mask)
    #             # weight_mask[getattr(self, "keep_current_channel_index"),:,:,:] = 1
    #             bias_mask[getattr(self, "keep_current_channel_index")] = 1

    #             # print(getattr(self, "keep_prev_channel_index"))
    #             # print(getattr(self, "keep_current_channel_index"))
    #             self.weight.mul_(weight_mask)
    #             self.bias.mul_(bias_mask)
                
    #             return
            
    #         else:
    #             weight = torch.index_select(self.weight, 1, getattr(self, "keep_prev_channel_index"))
    #             weight = torch.index_select(weight, 0, getattr(self, "keep_current_channel_index"))
    #             bias = torch.index_select(self.bias, 0, getattr(self, "keep_current_channel_index"))

    #             return weight, bias
            
    #     except KeyError as e:
    #         print("Unable to create prunning mask, prune the layer first!")
    #         raise e


    @torch.no_grad()
    def dynamic_quantize_per_tensor(self, bitwidth: int = 8):
        """Configure dynamic per-tensor quantization
        
        Args:
            bitwidth: Quantization bitwidth (<= 8)
        """
        assert bitwidth <= 8
        setattr(self, "quantization_type", DYNAMIC_QUANTIZATION_PER_TENSOR)
        setattr(self, "quantization_bitwidth", bitwidth)

        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        weight_scale = get_quantize_scale_per_tensor_sy(weight, bitwidth)
        weight_quant = quantize_per_tensor_sy(weight, weight_scale, bitwidth)

        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("weight_dmc", weight_quant)

    @torch.no_grad()
    def dynamic_quantize_per_channel(self, bitwidth: int = 8):
        """Configure dynamic per-channel quantization
        
        Args:
            bitwidth: Quantization bitwidth (<= 8)
        """
        assert bitwidth <= 8
        setattr(self, "quantization_type", DYNAMIC_QUANTIZATION_PER_CHANNEL)
        setattr(self, "quantization_bitwidth", bitwidth)

        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        weight_scale = get_quantize_scale_per_channel_sy(weight, bitwidth)
        weight_quant = quantize_per_channel_sy(weight, weight_scale, bitwidth)
        
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("weight_dmc", weight_quant)

    @torch.no_grad()
    def static_quantize_per_tensor(self,
                                 input_batch_real: torch.Tensor,
                                 input_batch_quant: torch.Tensor,
                                 input_scale: torch.Tensor,
                                 input_zero_point: torch.Tensor,
                                 bitwidth: int = 8):
        """Configure static per-tensor quantization
        
        Args:
            input_batch_real: FP32 input samples
            input_batch_quant: Quantized input samples
            input_scale: Input quantization scale
            input_zero_point: Input quantization zero point
            bitwidth: Quantization bitwidth (<= 8)
            
        Returns:
            Tuple of (real_output, quant_output, output_scale, output_zero_point)
        """
        assert bitwidth <= 8

        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_TENSOR)
        setattr(self, "quantization_bitwidth", bitwidth)

        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        # Quantize weights and biases
        weight_scale = get_quantize_scale_per_tensor_sy(weight, bitwidth)
        weight_quant = quantize_per_tensor_sy(weight, weight_scale, bitwidth)
        
        bias_quant = quantize_per_tensor_sy(
            bias, 
            (weight_scale * input_scale), 
            bitwidth * 4, 
            torch.int32
        )
        
        # Perform quantized convolution
        output_batch_quant_bias_dtype = torch.nn.functional.conv2d(
            (input_batch_quant.to(torch.int32) - input_zero_point),
            weight_quant.to(torch.int32),
            bias_quant.to(torch.int32)
        )

        try:
            output_batch_real = torch.nn.functional.conv2d(
                input_batch_real,
                weight,
                bias
            )
        except RuntimeError as e:
            print(e)
            print("This probably because the layer has been quantized before. "
                  "Multiple requantization not currently supported, try dynamic quantization.")
            raise RuntimeError

        # Calculate output quantization parameters
        output_scale, output_zero_point = get_quantize_scale_zero_point_per_tensor_assy(
            output_batch_real, 
            bitwidth
        )

        output_batch_quant = quantize_per_tensor_assy(
            output_batch_quant_bias_dtype * input_scale * weight_scale,
            output_scale,
            output_zero_point,
            bitwidth
        )

        # Register buffers for quantization parameters
        self.register_buffer("input_zero_point", input_zero_point)
        self.register_buffer("weight_dmc", weight_quant)
        self.register_buffer("bias_dmc", bias_quant)
        self.register_buffer("bias_scale", (weight_scale * input_scale))
        self.register_buffer("output_scale", output_scale)
        self.register_buffer("output_zero_point", output_zero_point)

        return output_batch_real, output_batch_quant, output_scale, output_zero_point

    @torch.no_grad()
    def static_quantize_per_channel(self,
                                  input_batch_real: torch.Tensor,
                                  input_batch_quant: torch.Tensor,
                                  input_scale: torch.Tensor,
                                  input_zero_point: torch.Tensor,
                                  bitwidth: int = 8):
        """Configure static per-channel quantization
        
        Args:
            Same as static_quantize_per_tensor
            
        Returns:
            Same as static_quantize_per_tensor
        """
        assert bitwidth <= 8

        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)
        setattr(self, "quantization_bitwidth", bitwidth)

        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        output_channel_axis = 1

        # Quantize weights and biases
        weight_scale = get_quantize_scale_per_channel_sy(weight, bitwidth)
        weight_quant = quantize_per_channel_sy(weight, weight_scale, bitwidth)

        bias_quant = quantize_per_channel_sy(
            bias, 
            (weight_scale * input_scale), 
            bitwidth * 4, 
            torch.int32
        )

        # Perform quantized convolution
        output_batch_quant_bias_dtype = torch.nn.functional.conv2d(
            (input_batch_quant.to(torch.int32) - input_zero_point),
            weight_quant.to(torch.int32),
            bias_quant.to(torch.int32)
        )

        try:
            output_batch_real = torch.nn.functional.conv2d(
                input_batch_real,
                weight,
                bias
            )
        except RuntimeError as e:
            print(e)
            print("This probably because the layer has been quantized before. "
                  "Multiple requantization not currently supported, try dynamic quantization.")
            raise RuntimeError

        # Calculate output quantization parameters
        output_scale, output_zero_point = get_quantize_scale_zero_point_per_tensor_assy(
            output_batch_real, 
            bitwidth
        )
        
        output_batch_quant = quantize_per_tensor_assy(
            output_batch_quant_bias_dtype * (input_scale * weight_scale).view(1, -1, 1, 1),
            output_scale,
            output_zero_point,
            bitwidth
        )
 
        # Register buffers for quantization parameters
        self.register_buffer("input_zero_point", input_zero_point)
        self.register_buffer("weight_dmc", weight_quant)
        self.register_buffer("bias_dmc", bias_quant)
        self.register_buffer("bias_scale", (weight_scale * input_scale))
        self.register_buffer("output_scale", output_scale)
        self.register_buffer("output_zero_point", output_zero_point)
        
        return output_batch_real, output_batch_quant, output_scale, output_zero_point


    # def get_output_tensor_shape(self, input_shape):

    #     C_in, H_in, W_in = input_shape
        
    #     # Unpack parameters (handle both int and tuple)
    #     def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
    #     # kH, kW = _pair(self.kernel_size)
    #     if self.bias is not None:
    #         C_out, _, kH, kW = self.get_compression_parameters()[0].size()
    #     else:
    #         C_out, _, kH, kW = self.get_compression_parameters().size()
            
    #     sH, sW = _pair(self.stride)
    #     dH, dW = _pair(self.dilation)

    #     pW = self.pad[0] + self.pad[1]
    #     pH = self.pad[2] + self.pad[3]
        
    #     H_out = ((H_in +  pH - dH * (kH - 1) - 1) // sH) + 1
    #     W_out = ((W_in +  pW - dW * (kW - 1) - 1) // sW) + 1
        
    #     return torch.Size((C_in, H_in +  pH, W_in +  pW)), torch.Size((C_out, H_out, W_out))
    

    # @torch.no_grad()
    # def convert_to_c(self, var_name, input_shape):
    #     """Generate C code declarations for this layer
        
    #     Args:
    #         var_name: Variable name to use in generated code
            
    #     Returns:
    #         Tuple of (header declaration, layer definition, parameter definition)
    #     """
    #     if self.bias is not None:
    #         weight, bias = self.get_compression_parameters()
    #         input_channel_size, input_row_size, input_col_size = input_shape

    #         output_channel_size, _,\
    #         kernel_row_size, kernel_col_size = weight.size()
    #         stride_row, stride_col = self.stride
    #         pad = self.pad
    #         groups = getattr(self, "prune_groups", self.groups)

    #         # Convert weights and parameters to C byte arrays
    #         param_header, param_def = convert_tensor_to_bytes_var(
    #             weight, 
    #             f"{var_name}_weight", 
    #             getattr(self, "quantization_bitwidth", 8)
    #         )
    #         layer_header = param_header
    #         layer_param_def = param_def

    #         param_header, param_def = convert_tensor_to_bytes_var(bias, f"{var_name}_bias")
    #         layer_header += param_header
    #         layer_param_def += param_def

    #         # Generate layer definition based on quantization mode
    #         if not hasattr(self, "quantization_type") or getattr(self, "quantization_type") == QUANTIZATION_NONE:
    #             layer_def = (
    #                 f"{self.__class__.__name__} {var_name}({input_channel_size}, "
    #                 f"{input_row_size}, {input_col_size}, {output_channel_size}, "
    #                 f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
    #                 "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
    #                 f"(float*){var_name}_weight, (float*){var_name}_bias);\n"
    #             )

    #     else:
    #         weight = self.get_compression_parameters()
    #         input_channel_size, input_row_size, input_col_size = input_shape

    #         output_channel_size, _,\
    #         kernel_row_size, kernel_col_size = weight.size()
    #         stride_row, stride_col = self.stride
    #         pad = self.pad
    #         groups = getattr(self, "prune_groups", self.groups)

    #         # Convert weights and parameters to C byte arrays
    #         param_header, param_def = convert_tensor_to_bytes_var(
    #             weight, 
    #             f"{var_name}_weight", 
    #             getattr(self, "quantization_bitwidth", 8)
    #         )
    #         layer_header = param_header
    #         layer_param_def = param_def

    #         # Generate layer definition based on quantization mode
    #         if not hasattr(self, "quantization_type") or getattr(self, "quantization_type") == QUANTIZATION_NONE:
    #             layer_def = (
    #                 f"{self.__class__.__name__} {var_name}({input_channel_size}, "
    #                 f"{input_row_size}, {input_col_size}, {output_channel_size}, "
    #                 f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
    #                 "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
    #                 f"(float*){var_name}_weight, nullptr);\n"
    #             )
    #     # Handle different quantization modes
    #     # if hasattr(self, "quantization_type"):
    #     #     if getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
    #     #         param_header, param_def = convert_tensor_to_bytes_var(
    #     #             self.weight_scale, 
    #     #             f"{var_name}_weight_scale"
    #     #         )
    #     #         layer_header += param_header
    #     #         layer_param_def += param_def
            
    #     #     elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
    #     #         param_header, param_def = convert_tensor_to_bytes_var(
    #     #             self.output_scale, 
    #     #             f"{var_name}_output_scale"
    #     #         )
    #     #         layer_header += param_header
    #     #         layer_param_def += param_def

    #     #         param_header, param_def = convert_tensor_to_bytes_var(
    #     #             self.output_zero_point, 
    #     #             f"{var_name}_output_zero_point"
    #     #         )
    #     #         layer_header += param_header
    #     #         layer_param_def += param_def

    #     #         param_header, param_def = convert_tensor_to_bytes_var(
    #     #             self.bias_scale, 
    #     #             f"{var_name}_bias_scale"
    #     #         )
    #     #         layer_header += param_header
    #     #         layer_param_def += param_def
   
    #     # Convert bias to C byte array
 
    #     # elif getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
    #     #     layer_def = (
    #     #         f"{self.__class__.__name__} {var_name}({input_channel_size}, "
    #     #         f"{input_row_size}, {input_col_size}, {output_channel_size}, "
    #     #         f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
    #     #         f"{padding}, (int8_t*){var_name}_weight, *(float*){var_name}_weight_scale, "
    #     #         f"(float*){var_name}_bias);\n"
    #     #     )
    #     # elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
    #     #     layer_def = (
    #     #         f"{self.__class__.__name__} {var_name}({input_channel_size}, "
    #     #         f"{input_row_size}, {input_col_size}, {output_channel_size}, "
    #     #         f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
    #     #         f"{padding}, {self.output_scale}, {self.output_zero_point}, "
    #     #         f"{self.input_zero_point}, (int8_t*){var_name}_weight, "
    #     #         f"(int32_t*){var_name}_bias, *(float*){var_name}_bias_scale);\n"
    #     #     )
    
    #     layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

    #     return layer_header, layer_def, layer_param_def