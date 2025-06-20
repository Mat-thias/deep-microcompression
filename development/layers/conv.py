"""
@file conv.py
@brief PyTorch implementation of Conv2d layer with support for:
    1. Standard floating-point operation
    2. Dynamic/static quantization (per-tensor and per-channel)
    3. Channel pruning
    4. C code generation for deployment
"""

__all__ = ["Conv2d"]

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


class Conv2d(nn.Conv2d):
    """Quantization-aware Conv2d layer with support for:
        - Standard convolution operation
        - Dynamic/static quantization (per-tensor and per-channel)
        - Channel pruning
        - C code generation for deployment
    """

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Output tensor after convolution with quantization if enabled
        """
        # Store input shape for later use in code generation
        setattr(self, "input_shape", input.size())

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

        # Perform convolution with appropriate padding
        if self.padding_mode != "zeros":
            input = nn.functional.conv2d(
                nn.functional.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        else:
            input = nn.functional.conv2d(
                input, weight, bias, self.stride, self.padding, self.dilation, self.groups
            )

        # Apply output quantization if in static quantization mode
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
                    input * self.bias_scale.view(1, -1, 1, 1), 
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
            metric: Importance metric ("l2" or other)
            
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
            keep_channel_index = None
            self.register_buffer("weight_dmc", weight_dmc)
            self.register_buffer("bias_dmc", bias)
            return keep_channel_index

        # Calculate channel importance
        importance = weight.pow(2) if metric == "l2" else weight.abs()
        channel_importance = importance.sum(dim=[1, 2, 3])
        threshold = channel_importance.quantile(sparsity)
        keep_channel_index = torch.nonzero(
            (channel_importance >= threshold).to(torch.int32)
        ).squeeze(dim=1)

        # Update weights and biases
        self.register_buffer("weight_dmc", torch.index_select(weight_dmc, 0, keep_channel_index))
        self.register_buffer("bias_dmc", torch.index_select(bias, 0, keep_channel_index))

        return keep_channel_index

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

        input_row_size, input_col_size = self.input_shape[2:4]

        output_channel_size, input_channel_size,\
        kernel_row_size, kernel_col_size = weight.size()
        stride_row, stride_col = self.stride
        padding = self.padding[0]

        # Convert weights and parameters to C byte arrays
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
   
        # Convert bias to C byte array
        param_header, param_def = convert_tensor_to_bytes_var(bias, f"{var_name}_bias")
        layer_header += param_header
        layer_param_def += param_def

        # Generate layer definition based on quantization mode
        if not hasattr(self, "quantization_type") or getattr(self, "quantization_type") == QUANTIZATION_NONE:
            layer_def = (
                f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                f"{padding}, (float*){var_name}_weight, (float*){var_name}_bias);\n"
            )
        elif getattr(self, "quantization_type") == DYNAMIC_QUANTIZATION_PER_TENSOR:
            layer_def = (
                f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                f"{padding}, (int8_t*){var_name}_weight, *(float*){var_name}_weight_scale, "
                f"(float*){var_name}_bias);\n"
            )
        elif getattr(self, "quantization_type") == STATIC_QUANTIZATION_PER_TENSOR:
            layer_def = (
                f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                f"{padding}, {self.output_scale}, {self.output_zero_point}, "
                f"{self.input_zero_point}, (int8_t*){var_name}_weight, "
                f"(int32_t*){var_name}_bias, *(float*){var_name}_bias_scale);\n"
            )
    
        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

        return layer_header, layer_def, layer_param_def