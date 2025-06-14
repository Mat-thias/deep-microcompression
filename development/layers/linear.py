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

from typing import Union

import torch
from torch import nn

from ..utilis import (
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
)


class Linear(nn.Linear):
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
        # Use quantized weights if available
        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        # Handle different quantization modes
        if hasattr(self, "quantization_type"):
            if getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
                weight = dequantize_per_tensor_sy(self.weight_dmc, self.weight_scale)
            elif getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_CHANNEL:
                weight = dequantize_per_channel_sy(self.weight_dmc, self.weight_scale)
            elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
                input = input.to(torch.int32) - self.input_zero_point
                weight = self.weight_dmc.to(torch.int32)
                bias = self.bias_dmc
            elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
                input = input.to(torch.int32) - self.input_zero_point
                weight = self.weight_dmc.to(torch.int32)
                bias = self.bias_dmc

        # Perform linear operation
        input = nn.functional.linear(input, weight, bias)

        # Post-quantization if needed
        if hasattr(self, "quantization_type"):
            if getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
                input = quantize_per_tensor_assy(
                    input * self.bias_scale, 
                    self.output_scale, 
                    self.output_zero_point, 
                    self.quantization_bitwidth
                )
            elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_CHANNEL:
                input = quantize_per_tensor_assy(
                    input * self.bias_scale.view(1, -1), 
                    self.output_scale, 
                    self.output_zero_point, 
                    self.quantization_bitwidth
                )
        return input

    @torch.no_grad()
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Union[torch.Tensor, None], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        """Prune channels based on importance metric
        
        Args:
            sparsity: Target sparsity ratio (0-1)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Importance metric ("l2" or absolute value)
            
        Returns:
            Indices of kept channels
        """
        sparsity = min(max(0., sparsity), 1.)

        weight = self.weight_dmc if hasattr(self, "weight_dmc") else self.weight
        bias = self.bias_dmc if hasattr(self, "bias_dmc") else self.bias

        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_channels)
        weight_dmc = torch.index_select(weight, 1, keep_prev_channel_index)

        if is_output_layer:
            # Skip pruning for output layer
            keep_channel_index = None
            self.register_buffer("weight_dmc", weight_dmc)
            self.register_buffer("bias_dmc", bias)
            return keep_channel_index

        # Calculate channel importance
        importance = weight.pow(2) if metric == "l2" else weight.abs()
        channel_importance = importance.sum(dim=[1])
        threshold = channel_importance.quantile(sparsity)
        keep_channel_index = torch.nonzero(
            (channel_importance >= threshold).to(torch.int32)
        ).squeeze(dim=1)

        # Update weights and biases
        self.register_buffer(
            "weight_dmc", 
            torch.index_select(weight_dmc, 0, keep_channel_index)
        )
        self.register_buffer(
            "bias_dmc", 
            torch.index_select(bias, 0, keep_channel_index)
        )

        return keep_channel_index

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

    @torch.no_grad()
    def convert_to_c(self, var_name):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        weight = getattr(self, "weight_dmc", self.weight)
        bias = getattr(self, "bias_dmc", self.bias)
        output_feature_size, input_feature_size = weight.size()

        # Convert weights to C representation
        param_header, param_def = convert_tensor_to_bytes_var(
            weight, 
            f"{var_name}_weight", 
            getattr(self, "quantization_bitwidth", 8)
        )
        layer_header = param_header
        layer_param_def = param_def

        # Handle different quantization modes
        if hasattr(self, "quantization_type"):
            if getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
                param_header, param_def = convert_tensor_to_bytes_var(
                    self.weight_scale, 
                    f"{var_name}_weight_scale"
                )
                layer_header += param_header
                layer_param_def += param_def
            elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
                param_header, param_def = convert_tensor_to_bytes_var(
                    self.output_scale, 
                    f"{var_name}_output_scale"
                )
                layer_header += param_header
                layer_param_def += param_def

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.output_zero_point, 
                    f"{var_name}_output_zero_point"
                )
                layer_header += param_header
                layer_param_def += param_def

                param_header, param_def = convert_tensor_to_bytes_var(
                    self.bias_scale, 
                    f"{var_name}_bias_scale"
                )
                layer_header += param_header
                layer_param_def += param_def

        # Convert biases to C representation
        param_header, param_def = convert_tensor_to_bytes_var(
            bias, 
            f"{var_name}_bias"
        )
        layer_header += param_header
        layer_param_def += param_def

        # Generate layer definition based on quantization mode
        if not hasattr(self, "quantization_type") or getattr(self, "quantization_type", QUANTIZATION_NONE) == QUANTIZATION_NONE:
            layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, (float*){var_name}_bias);\n"
        elif getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
            layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, *(float*){var_name}_weight_scale, (float*){var_name}_bias);\n"
        elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
            layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, {self.output_scale}, {self.output_zero_point}, {self.input_zero_point}, (int8_t*){var_name}_weight, (int32_t*){var_name}_bias, *(float*){var_name}_bias_scale);\n"
        
        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

        return layer_header, layer_def, layer_param_def