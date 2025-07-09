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

    get_size_in_bits,

    QUANTIZATION_NONE,
    DYNAMIC_QUANTIZATION_PER_TENSOR,
    DYNAMIC_QUANTIZATION_PER_CHANNEL,
    STATIC_QUANTIZATION_PER_TENSOR,
    STATIC_QUANTIZATION_PER_CHANNEL,
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

    def forward(self, input):
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Output tensor according to current quantization mode
        """
        input = super().forward(input)
        return input

    
    @torch.no_grad()
    def prepare_prune_channel(
            self, 
            sparsity: float, 
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
        super().prepare_prune_channel()

        sparsity = min(max(0., sparsity), 1.)

        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_features)

        if is_output_layer:
            # Skip pruning for output layer
            keep_current_channel_index = torch.arange(self.out_features)
        else:

            # Calculate channel importance
            importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
            channel_importance = importance.sum(dim=[1])
            threshold = channel_importance.quantile(sparsity)
            keep_current_channel_index = torch.nonzero(
                (channel_importance >= threshold).to(torch.int32)
            ).squeeze(dim=1)
            setattr(self, "keep_current_channel_index", keep_current_channel_index)
            
        self.__dict__["_dmc"]["prune_channel"] = dict()
        self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"] = keep_prev_channel_index
        self.__dict__["_dmc"]["prune_channel"]["keep_current_channel_index"] = keep_current_channel_index

        return keep_current_channel_index
    

    @torch.no_grad()
    def apply_prune_channel(self):
        super().apply_prune_channel()

        weight_mask_prev_channel = torch.zeros_like(self.weight)
        weight_mask_current_channel = torch.zeros_like(self.weight)

        keep_prev_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_prev_channel_index"]
        keep_current_channel_index = self.__dict__["_dmc"]["prune_channel"]["keep_current_channel_index"]

        weight_mask_prev_channel[:, keep_prev_channel_index] = 1
        weight_mask_current_channel[keep_current_channel_index, :] = 1
        weight_mask = torch.mul(weight_mask_prev_channel, weight_mask_current_channel)
        
        self.weight.mul_(weight_mask)
    
        if self.bias is not None:
            bias_mask = torch.zeros_like(self.bias)
            bias_mask[keep_current_channel_index] = 1
            
            self.bias.mul_(bias_mask)

        return
            
            
        
    def apply_prune_channel_external(self, weight, bias=None):

        super().apply_prune_channel()
        
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
        q_type,
    ):
        super().prepare_quantization(bitwidth, q_type)

        if q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
            self.prepare_dynamic_quantization_per_tensor(bitwidth)


    def apply_quantization(self):
        super().apply_quantization()
        
        if self.__dict__["_dmc"]["quantization"]["bitwidth"] is None:
            return

        q_type = self.__dict__["_dmc"]["quantization"]["type"]                     
        if q_type == DYNAMIC_QUANTIZATION_PER_TENSOR:
            self.apply_dynamic_quantization_per_tensor() 

        return


    def apply_quantization_external(self, weight, bias=None):
          
        if "quantization" not in self.__dict__["_dmc"]:
            raise AttributeError("Layer must be prepared before applying compression")
        
        type = self.__dict__["_dmc"]["quantization"]["type"]                     

        if type == DYNAMIC_QUANTIZATION_PER_TENSOR:
            return self.apply_dynamic_quantization_per_tensor_external(weight), bias 


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

    


    @torch.no_grad()
    def get_size_in_bits(self):
        
        if self.bias is not None:
            weight, bias = self.get_compression_parameters()
        else:
            weight = self.get_compression_parameters()

        is_packed = False
        bitwidth = None
        if "quantization" in self.__dict__["_dmc"]:
            is_packed = True
            bitwidth = self.__dict__["_dmc"]["quantization"]["bitwidth"]


        size = 0
        size += get_size_in_bits(weight, is_packed=is_packed, bitwidth=bitwidth)

        if self.bias is not None:
            size += get_size_in_bits(bias, is_packed=is_packed, bitwidth=bitwidth)
            
        return size



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

        if "prune_channel" in self.__dict__["_dmc"]:
            weight = self.apply_prune_channel_external(weight)
        if "quantization" in self.__dict__["_dmc"]:
            weight = self.apply_dynamic_quantization_per_tensor_external(weight)

        return weight



    def get_output_tensor_shape(self, input_shape):
        
        if self.bias is not None:
            out_features, _ = self.get_compression_parameters()[0].size()
        else:
            out_features, _ = self.get_compression_parameters().size()
        return torch.Size((out_features,)), torch.Size((out_features,))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        # """

        if self.bias is not None:
            weight, bias = self.get_compression_parameters()
            
            output_feature_size, input_feature_size = weight.size()

            # Convert weights to C representation
            param_header, param_def = convert_tensor_to_bytes_var(
                weight, 
                f"{var_name}_weight", 
                getattr(self, "quantization_bitwidth", 8)
            )   
            layer_header = param_header
            layer_param_def = param_def

            param_header, param_def = convert_tensor_to_bytes_var(
                bias, 
                f"{var_name}_bias"
            )
            layer_header += param_header
            layer_param_def += param_def

            # Generate layer definition based on quantization mode
            if not hasattr(self, "quantization_type") or getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, (float*){var_name}_bias);\n"
        else:
            weight = self.get_compression_parameters()
            
            output_feature_size, input_feature_size = weight.size()

            # Convert weights to C representation
            param_header, param_def = convert_tensor_to_bytes_var(
                weight, 
                f"{var_name}_weight", 
                getattr(self, "quantization_bitwidth", 8)
            )   
            layer_header = param_header
            layer_param_def = param_def

            # Generate layer definition based on quantization mode
            if not hasattr(self, "quantization_type") or getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, nullptr);\n"
        
        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

        return layer_header, layer_def, layer_param_def






















    # @torch.no_grad()
    # def apply_prune_channel(self, inplace:bool = True) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    #     weight_mask_prev_channel = torch.zeros_like(self.weight)
    #     weight_mask_current_channel = torch.zeros_like(self.weight)
    #     bias_mask = torch.zeros_like(self.bias)

    #     try:
    #         if inplace:
    #             weight_mask_prev_channel[:,getattr(self, "keep_prev_channel_index")] = 1
    #             weight_mask_current_channel[getattr(self, "keep_current_channel_index"),:] = 1
    #             weight_mask = torch.mul(weight_mask_prev_channel, weight_mask_current_channel)
            
    #             # weight_mask[:,getattr(self, "keep_prev_channel_index")] = 1
    #             # weight_mask[getattr(self, "keep_current_channel_index")] = 1
    #             bias_mask[getattr(self, "keep_current_channel_index")] = 1
                

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
        assert bitwidth <= 8, "Bitwidth should be less than 8"
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
        assert bitwidth <= 8, "Bitwidth should be less than 8"
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
        assert bitwidth <= 8, "Bitwidth should be less than 8"

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
        
        # Forward pass in quantized domain
        output_batch_quant_bias_dtype = torch.nn.functional.linear(
            (input_batch_quant.to(torch.int32) - input_zero_point),
            weight_quant.to(torch.int32),
            bias_quant.to(torch.int32)
        )

        try:
            # Forward pass in real domain
            output_batch_real = torch.nn.functional.linear(
                input_batch_real,
                weight,
                bias
            )
        except RuntimeError as e:
            print(e)
            print("This probably because the layer has been quantized before, "
                  "Multiple requantization no currently supported, try dynamic quantization.")
            raise RuntimeError

        # Calculate output quantization parameters
        output_scale, output_zero_point = get_quantize_scale_zero_point_per_tensor_assy(
            output_batch_real, 
            bitwidth
        )
        
        # Quantize output
        output_batch_quant = quantize_per_tensor_assy(
            output_batch_quant_bias_dtype * input_scale * weight_scale,
            output_scale,
            output_zero_point,
            bitwidth
        )

        # Register buffers
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
        assert bitwidth <= 8, "Bitwidth should be less than 8"

        setattr(self, "quantization_type", STATIC_QUANTIZATION_PER_CHANNEL)
        setattr(self, "quantization_bitwidth", bitwidth)

        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        # Quantize weights and biases
        weight_scale = get_quantize_scale_per_channel_sy(weight, bitwidth)
        weight_quant = quantize_per_channel_sy(weight, weight_scale, bitwidth)
        bias_quant = quantize_per_channel_sy(
            bias, 
            (weight_scale * input_scale), 
            bitwidth * 4, 
            torch.int32
        )
    
        # Forward pass in quantized domain
        output_batch_quant_bias_dtype = torch.nn.functional.linear(
            (input_batch_quant.to(torch.int32) - input_zero_point),
            weight_quant.to(torch.int32),
            bias_quant.to(torch.int32)
        )

        try:
            # Forward pass in real domain
            output_batch_real = torch.nn.functional.linear(
                input_batch_real,
                weight,
                bias
            )
        except RuntimeError as e:
            print(e)
            print("This probably because the layer has been quantized before, "
                  "Multiple requantization no currently supported, try dynamic quantization.")
            raise RuntimeError

        # Calculate output quantization parameters
        output_scale, output_zero_point = get_quantize_scale_zero_point_per_tensor_assy(
            output_batch_real, 
            bitwidth
        )

        # Quantize output
        output_batch_quant = quantize_per_tensor_assy(
            output_batch_quant_bias_dtype * input_scale * weight_scale,
            output_scale,
            output_zero_point,
            bitwidth
        )

        # Register buffers
        self.register_buffer("input_zero_point", input_zero_point)
        self.register_buffer("weight_dmc", weight_quant)
        self.register_buffer("bias_dmc", bias_quant)
        self.register_buffer("bias_scale", (weight_scale * input_scale))
        self.register_buffer("output_scale", output_scale)
        self.register_buffer("output_zero_point", output_zero_point)

        return output_batch_real, output_batch_quant, output_scale, output_zero_point


    # def get_output_tensor_shape(self, input_shape):
        
    #     if self.bias is not None:
    #         out_features, _ = self.get_compression_parameters()[0].size()
    #     else:
    #         out_features, _ = self.get_compression_parameters().size()
    #     return torch.Size((out_features,)), torch.Size((out_features,))
    

    # @torch.no_grad()
    # def convert_to_c(self, var_name, input_shape):
    #     """Generate C code declarations for this layer
        
    #     Args:
    #         var_name: Variable name to use in generated code
            
    #     Returns:
    #         Tuple of (header declaration, layer definition, parameter definition)
    #     # """
    #     # weight = getattr(self, "weight_dmc", self.weight)
    #     # bias = getattr(self, "bias_dmc", self.bias)

    #     if self.bias is not None:
    #         weight, bias = self.get_compression_parameters()
            
    #         output_feature_size, input_feature_size = weight.size()

    #         # Convert weights to C representation
    #         param_header, param_def = convert_tensor_to_bytes_var(
    #             weight, 
    #             f"{var_name}_weight", 
    #             getattr(self, "quantization_bitwidth", 8)
    #         )   
    #         layer_header = param_header
    #         layer_param_def = param_def

    #         param_header, param_def = convert_tensor_to_bytes_var(
    #             bias, 
    #             f"{var_name}_bias"
    #         )
    #         layer_header += param_header
    #         layer_param_def += param_def

    #         # Generate layer definition based on quantization mode
    #         if not hasattr(self, "quantization_type") or getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
    #             layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, (float*){var_name}_bias);\n"
    #     else:
    #         weight = self.get_compression_parameters()
            
    #         output_feature_size, input_feature_size = weight.size()

    #         # Convert weights to C representation
    #         param_header, param_def = convert_tensor_to_bytes_var(
    #             weight, 
    #             f"{var_name}_weight", 
    #             getattr(self, "quantization_bitwidth", 8)
    #         )   
    #         layer_header = param_header
    #         layer_param_def = param_def

    #         # Generate layer definition based on quantization mode
    #         if not hasattr(self, "quantization_type") or getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
    #             layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, nullptr);\n"


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

    #     # Convert biases to C representation
    #     # param_header, param_def = convert_tensor_to_bytes_var(
    #     #     bias, 
    #     #     f"{var_name}_bias"
    #     # )
    #     # layer_header += param_header
    #     # layer_param_def += param_def

    #     # # Generate layer definition based on quantization mode
    #     # if not hasattr(self, "quantization_type") or getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
    #     #     layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, (float*){var_name}_bias);\n"
    #     # elif getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
    #     #     layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, *(float*){var_name}_weight_scale, (float*){var_name}_bias);\n"
    #     # elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
    #     #     layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, {self.output_scale}, {self.output_zero_point}, {self.input_zero_point}, (int8_t*){var_name}_weight, (int32_t*){var_name}_bias, *(float*){var_name}_bias_scale);\n"
        
    #     layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

    #     return layer_header, layer_def, layer_param_def