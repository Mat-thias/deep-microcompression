from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Callable
from math import prod

import torch
from torch import nn

from ..utils import (

    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,

    get_quantize_scale_per_tensor_sy,
    get_quantize_scale_zero_point_per_tensor_assy,
    get_quantize_scale_per_channel_sy, 
    get_quantize_scale_zero_point_per_channel_assy,

    fake_quantize_per_tensor_sy,
    fake_quantize_per_tensor_assy,
    fake_quantize_per_channel_sy, 
    fake_quantize_per_channel_assy,

    quantize_per_tensor_assy,
    quantize_per_tensor_sy,
    quantize_per_channel_assy, 
    quantize_per_channel_sy
)


class Layer(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):

        setattr(self, "_dmc", dict())
        super().__init__(*args, **kwargs)
        self.is_pruned_channel = False
        self.is_quantized = False

    @property
    def is_compressed(self):
        return self.is_pruned_channel or self.is_quantized

    @abstractmethod
    def init_prune_channel(self):
        pass

    @abstractmethod
    def init_quantize(self, q_type, bitwidth):
        pass


    @abstractmethod
    def get_compression_parameters(self):
        pass


    @abstractmethod
    def get_size_in_bits(self):
        pass

    @abstractmethod
    def get_output_tensor_shape(self, input_shape):
        print("called layer")

    @abstractmethod
    def convert_to_c(self, var_name):
        pass




class Quantize:

    def __init__(
        self, 
        module: Layer, 
        bitwidth: int, 
        scheme: QuantizationScheme, 
        granularity: QuantizationGranularity, 
        scale_type: QuantizationScaleType, 
        base: Optional[Iterable["Quantize"]] = None, 
        base_accumulator: Optional[Callable[[torch.Tensor, int, Iterable["Quantize"]], torch.Tensor]] = None
    ) -> None:

        self.module = module
        self.bitwidth = bitwidth
        self.scheme = scheme
        self.scale_type = scale_type
        self.granularity = granularity

        self.scale = None
        self.zero_point = None

        self.base = base

        if base is not None:
            if base_accumulator is None:
                if scale_type == QuantizationScaleType.ASSYMMETRIC:
                    print(base)
                    self.base_accumulator = lambda x, bitdwidth, base : prod([b.scale for b in base]), sum([b.zero_point for b in base])
                else:
                    self.base_accumulator = lambda x, bitdwidth, base : prod([b.scale for b in base])
            else:
                self.base_accumulator = base_accumulator


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.module.training:
            self.update_parameters(x)
        return self.fake_apply(x)
                 
 
    def update_parameters(self, x: torch.Tensor) -> None:
        if self.base is None:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                self.scale = get_quantize_scale_per_tensor_sy(x, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                             get_quantize_scale_per_channel_sy(x, self.bitwidth)
            else:
                self.scale, self.zero_point = get_quantize_scale_zero_point_per_tensor_assy(x, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                                              get_quantize_scale_zero_point_per_channel_assy(x, self.bitwidth)
        else:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                self.scale = self.base_accumulator(x, self.bitwidth, self.base)    
            else:
                self.scale, self.zero_point = self.base_accumulator(x, self.bitwidth, self.base)    
                

    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return fake_quantize_per_tensor_sy(x, self.scale, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   fake_quantize_per_channel_sy(x, self.scale, self.bitwidth)
        return fake_quantize_per_tensor_assy(x, self.scale, self.zero_point, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               fake_quantize_per_channel_assy(x, self.scale, self.zero_point, self.bitwidth)

    def apply(self, x: torch.Tensor, prune_channel: Optional["Prune_Channel"] = None) -> torch.Tensor:
        dtype = torch.int32 if self.bitwidth > 8 else torch.int8
        
        scale = self.scale
        zero_point = self.zero_point

        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            scale = prune_channel.apply(scale)
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: zero_point = prune_channel.apply(zero_point) 
        
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return quantize_per_tensor_sy(x, scale, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   quantize_per_channel_sy(x, scale, self.bitwidth, dtype=dtype)
        return quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               quantize_per_channel_assy(x, scale, zero_point, self.bitwidth, dtype=dtype)
    

class Prune_Channel:

    def __init__(
        self, 
        module: Layer, 
        keep_current_channel_index: torch.Tensor, 
        keep_prev_channel_index: Optional[torch.Tensor]=None
    ) -> None:
        self.module = module
        self.keep_current_channel_index = keep_current_channel_index
        self.keep_prev_channel_index = keep_prev_channel_index


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.module.training:
            self.update_parameters(x)
        return self.fake_apply(x)
    
    def update_parameters(self, x: torch.Tensor) -> None:
        pass


    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
    
        if x.ndim > 1:
            mask_prev_channel = torch.zeros_like(x)
            mask_current_channel = torch.zeros_like(x)

            mask_prev_channel_index = [slice(None)]*x.ndim
            mask_prev_channel_index[1] = self.keep_prev_channel_index

            mask_current_channel_index = [slice(None)]*x.ndim
            mask_current_channel_index[0] = self.keep_current_channel_index

            mask_prev_channel[mask_prev_channel_index] = 1 
            mask_current_channel[mask_current_channel_index] = 1 

            mask = torch.mul(mask_current_channel, mask_prev_channel)
        else:
            mask = torch.zeros_like(x)
            mask[self.keep_current_channel_index] = 1

        return torch.mul(x, mask)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 1:
            x = torch.index_select(x, 1, self.keep_prev_channel_index)
            x = torch.index_select(x, 0, self.keep_current_channel_index)
        else:
            x = torch.index_select(x, 0, self.keep_current_channel_index)
        return x