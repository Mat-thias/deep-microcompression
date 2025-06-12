import struct

import torch
from torch import nn


QUANTIZATION_NONE = 0
DYNAMIC_QUANTIZATION_PER_TENSOR = 1
DYNAMIC_QUANTIZATION_PER_CHANNEL = 2
STATIC_QUANTIZATION_PER_TENSOR = 3
STATIC_QUANTIZATION_PER_CHANNEL = 4

INT8_BYTE_PER_LINE = 16
FLOAT32_BYTE_PER_LINE = 4
INT32_BYTE_PER_LINE = 4

def get_bitwidth_range(bitwidth):

    qmin = -(2 ** (bitwidth - 1))
    qmax = (2 ** (bitwidth - 1)) - 1

    return qmin, qmax

def get_quantize_scale_per_tensor_sy(tensor_real, bitwidth:int = 8, metric:str = "l2"):
    _, qmax = get_bitwidth_range(bitwidth)
    rmax = tensor_real.abs().max()
    scale =  (rmax / qmax)

    return scale

def get_quantize_scale_zero_point_per_tensor_assy(tensor_real, bitwidth:int = 8, metric:str = "l2"):

    qmin, qmax = get_bitwidth_range(bitwidth)

    rmin = tensor_real.min()
    rmax = tensor_real.max()

    scale =  (rmax - rmin )/ (qmax - qmin)
    zero_point = torch.round(qmin - (rmin / scale)).to(torch.int8)

    return scale, zero_point

def get_quantize_scale_per_channel_sy(tensor_real, bitwidth:int = 8, metric:str = "l2"):
    _, qmax = get_bitwidth_range(bitwidth)
    rmax = tensor_real.abs().view(tensor_real.size(0), -1).max(dim=1)[0]
    scale =  (rmax / qmax)

    return scale

def get_quantize_scale_zero_point_per_channel_sy(tensor_real, bitwidth:int = 8, metric:str = "l2"):
    qmin, qmax = get_bitwidth_range(bitwidth)

    rmin = tensor_real.view(tensor_real.size(0), -1).min(dim=1)[0]
    rmax = tensor_real.view(tensor_real.size(0), -1).max(dim=1)[0]

    scale =  (rmax - rmin) / (qmax - qmin)
    zero_point = torch.round(qmin - (rmin / scale)).to(torch.int8)

    return scale, zero_point



def quantize_per_tensor_sy(tensor_real, scale:torch.Tensor, bitwidth:int = 8, dtype=torch.int8):
    _, qmax = get_bitwidth_range(bitwidth)
    return torch.clamp(
        torch.round(tensor_real / scale), -qmax, qmax
    ).to(dtype)


def dequantize_per_tensor_sy(tensor_quant, scale:torch.Tensor):
    return tensor_quant * scale



def quantize_per_tensor_assy(tensor_real, scale:torch.Tensor, zero_point:torch.Tensor, bitwidth:int = 8, dtype=torch.int8):

    qmin, qmax = get_bitwidth_range(bitwidth)

    return torch.clamp(
        torch.round(tensor_real / scale) + zero_point, qmin, qmax
    ).to(dtype)

def dequantize_per_tensor_assy(tensor_quant, scale:torch.Tensor, zero_point:torch.Tensor):
    return (tensor_quant - zero_point) * scale 


def quantize_per_channel_sy(tensor_real, scale:torch.Tensor, bitwidth:int = 8, dtype=torch.int8):

    _, qmax = get_bitwidth_range(bitwidth)

    broadcast_shape = [1] * tensor_real.ndim
    broadcast_shape[0] = -1

    return torch.clamp(
        torch.round(tensor_real / scale.view(*broadcast_shape)), -qmax, qmax
    ).to(dtype)

def dequantize_per_channel_sy(tensor_quant, scale:torch.Tensor):
    broadcast_shape = [1] * tensor_quant.ndim
    broadcast_shape[0] = -1

    return tensor_quant * scale.view(*broadcast_shape)



def quantize_per_channel_assy(tensor_real, scale:torch.Tensor, zero_point:torch.Tensor, bitwidth:int = 8, dtype=torch.int8):

    qmin, qmax = get_bitwidth_range(bitwidth)

    broadcast_shape = [1] * tensor_real.ndim
    broadcast_shape[0] = -1

    return torch.clamp(
        torch.round(tensor_real / scale.view(*broadcast_shape)) + zero_point.view(*broadcast_shape), qmin, qmax
    ).to(dtype)

def dequantize_per_channel_assy(tensor_quant, scale:torch.Tensor, zero_point:torch.Tensor):
    broadcast_shape = [1] * tensor_quant.ndim
    broadcast_shape[0] = -1

    return (tensor_quant - zero_point.view(*broadcast_shape)) * scale.view(*broadcast_shape)



def float32_to_bytes(val):
    return list(struct.pack("<f", val))

def int32_to_bytes(val):
    return list(struct.pack("<i", val))  # Little endian int32


def int8_to_bytes(val):
    # return list(struct.pack("<i", val))  # Little endian int8
    return list(struct.pack("<b", val))

def int4_to_bytes(data):
    byte = 0
    for val in data[::-1]:
        byte = (byte << 4 | val & 0x0F)
    
    return list(struct.pack("<b", byte))


def convert_tensor_to_bytes_var(tensor:torch.Tensor, var_name, bitwidth:int=8):

    if tensor.dtype == torch.float:
        byte_convert = float32_to_bytes
        byte_per_line = FLOAT32_BYTE_PER_LINE

    elif tensor.dtype == torch.int32:
        byte_convert = int32_to_bytes
        byte_per_line = INT32_BYTE_PER_LINE
    else:
        byte_convert = int8_to_bytes
        byte_per_line = INT8_BYTE_PER_LINE

    var_header_str = f"extern const uint8_t {var_name}[];\n"
    var_def_str = f"\nconst uint8_t {var_name}[] = {{\n"

    if tensor.dtype != torch.int8 or bitwidth == 8:
        for line in torch.split(tensor.flatten(), byte_per_line):
            var_def_str += "    " + ", ".join(
                [f"0x{b:02X}" for val in line for b in byte_convert(val)]
            ) + ",\n"
    else:
        data_per_byte = 8 // bitwidth
        tensor = tensor.flatten()


        if bitwidth == 4:
            byte_convert = int4_to_bytes
        elif bitwidth == 2:
            byte_convert = int2_to_bytes
        else:
            raise RuntimeError(f"Conversion of model to quantized {bitwidth} bitwidth not support with packing!!")

        for line in torch.split(tensor.flatten(), INT8_BYTE_PER_LINE * data_per_byte):
            bytes = []

            for i in range(len(line)//data_per_byte):
                data = []

                for pos in range(data_per_byte):
                    data.append(line[(i*data_per_byte)+pos])
                bytes.append(byte_convert(data))

            if len(line)%data_per_byte != 0:
                data = []
                i += 1
                for pos in range(len(line)%data_per_byte):
                    data.append(line[(i*data_per_byte)+pos])
                    bytes.append(byte_convert(data))

            var_def_str += "    " + ", ".join(
                [f"0x{b:02X}" for val in bytes for b in val]
            ) + ",\n"
            
    var_def_str += "};\n"

    return var_header_str, var_def_str


