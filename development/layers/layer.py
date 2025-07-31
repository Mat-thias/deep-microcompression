from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Callable, Tuple
from math import prod

import torch
from torch import nn

# from ..models.sequential import Sequential

from ..utils import (

    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,

    get_quantize_scale_sy,
    get_quantize_scale_zero_point_assy,

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

    def is_prunable(self):
        return bool(self.get_prune_channel_possible_hypermeters())

    @abstractmethod
    def get_prune_channel_possible_hypermeters(self):
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
        pass

    @abstractmethod
    def convert_to_c(self, var_name):
        pass




class Quantize:

    def __init__(
        self, 
        module,#: Sequential, 
        bitwidth: int, 
        scheme: QuantizationScheme, 
        granularity: QuantizationGranularity, 
        scale_type: QuantizationScaleType, 
        avg_exp: float = 0.1,
        base: Optional[Iterable["Quantize"]] = None, 
        base_accumulator: Optional[Callable[[torch.Tensor, int, Iterable["Quantize"]], torch.Tensor]] = None,
        prune_channel: Optional["Prune_Channel"] = None
    ) -> None:

        self.module = module
        self.bitwidth = bitwidth
        self.scheme = scheme
        self.granularity = granularity
        self.scale_type = scale_type
        self.avg_exp = avg_exp
        self.rmin = None
        self.rmax = None 

        self.base = base

        if base is not None:
            if base_accumulator is None:
                if scale_type == QuantizationScaleType.ASSYMMETRIC:
                    self.base_accumulator: Callable[[Iterable["Quantize"]], Tuple[torch.Tensor, torch.Tensor]] = lambda base : prod([b.scale for b in base]), sum([b.zero_point for b in base])
                else:
                    self.base_accumulator: Callable[[Iterable["Quantize"]], torch.Tensor] = lambda base : prod([b.scale for b in base])
            else:
                self.base_accumulator = base_accumulator

        self.prune_channel = prune_channel

    @property
    def scale(self):
        if self.base is None:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                scale = get_quantize_scale_sy(self.rmax, self.bitwidth)
            else:
                scale = get_quantize_scale_zero_point_assy(self.rmax, self.rmin, self.bitwidth)[0]
        else:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                scale = self.base_accumulator(self.base)    
            else:
                scale = self.base_accumulator(self.base)[0]
        return scale
    
    @property
    def zero_point(self):
        assert self.scale_type == QuantizationScaleType.ASSYMMETRIC, f"scale type should be {QuantizationScaleType.ASSYMMETRIC}"
        if self.base is None:
            return get_quantize_scale_zero_point_assy(self.rmax, self.rmin, self.bitwidth)[1]
        else:
            return self.base_accumulator(self.base)[1]


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.module.training:
            self.update_parameters(x)
        return self.fake_apply(x)
 
    @torch.no_grad()
    def update_parameters(self, x: torch.Tensor) -> None:
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            if self.granularity == QuantizationGranularity.PER_TENSOR:
                if self.rmax is None: self.rmax = x.abs().max()
                else: self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.abs().max() 
            else:
                if self.rmax is None: self.rmax = x.abs().view(x.size(0), -1).max(dim=1)[0]
                else: self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.abs().view(x.size(0), -1).max(dim=1)[0]
            # self.rmax = x.max() if self.granularity == QuantizationGranularity.PER_TENSOR else x.abs().view(x.size(0), -1).max(dim=1)[0]
        else:
            if self.granularity == QuantizationGranularity.PER_TENSOR:
                if self.rmax is None or self.rmin is None: 
                    self.rmax = x.max()
                    self.rmin = x.min()
                else:
                    self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.max()
                    self.rmin = self.rmin * (1 -self.avg_exp) + self.avg_exp * x.min()
            else:
                if self.rmax is None or self.rmin is None: 
                    self.rmax = x.view(x.size(0), -1).max(dim=1)[0]
                    self.rmin = x.view(x.size(0), -1).min(dim=1)[0]
                else:
                    self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.view(x.size(0), -1).max(dim=1)[0]
                    self.rmin = self.rmin * (1 -self.avg_exp) + self.avg_exp * x.view(x.size(0), -1).min(dim=1)[0]
            # self.rmax = x.max() if self.granularity == QuantizationGranularity.PER_TENSOR else x.view(x.size(0), -1).max(dim=1)[0]
            # self.rmin = x.min() if self.granularity == QuantizationGranularity.PER_TENSOR else x.view(x.size(0), -1).min(dim=1)[0]
        # print(self.scale, self.module, self.granularity, self.scale_type, "herrrrkkkkkk")
        # print(self.rmax, self.module, self.granularity, self.scale_type, "herrrrkkkkkk")

        # if self.base is None:
        #     if self.scale_type == QuantizationScaleType.SYMMETRIC:
        #         self.rmax = x.max()
        #         # self.scale = get_quantize_scale_per_tensor_sy(x, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
        #         #              get_quantize_scale_per_channel_sy(x, self.bitwidth)
        #     else:
        #         self.rmax = x.max()
        #         self.rmin = x.min()
        #         # self.scale, self.zero_point = get_quantize_scale_zero_point_per_tensor_assy(x, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
        #         #                               get_quantize_scale_zero_point_per_channel_assy(x, self.bitwidth)
        # else:
        #     if self.scale_type == QuantizationScaleType.SYMMETRIC:
        #         self.scale = self.base_accumulator(x, self.bitwidth, self.base)    
        #     else:
        #         self.scale, self.zero_point = self.base_accumulator(x, self.bitwidth, self.base)    
                

    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
               
        scale = self.scale
        if self.scale_type == QuantizationScaleType.ASSYMMETRIC:
            zero_point = self.zero_point

        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            scale = self.prune_channel.fake_apply(scale)
            scale[scale == 0] = 1
            
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: 
                zero_point = self.prune_channel.fake_apply(zero_point) 
        
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return fake_quantize_per_tensor_sy(x, scale, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   fake_quantize_per_channel_sy(x, scale, self.bitwidth)
        return fake_quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               fake_quantize_per_channel_assy(x, scale, zero_point, self.bitwidth)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.int32 if self.bitwidth > 8 else torch.int8
        
        scale = self.scale
        if self.scale_type == QuantizationScaleType.ASSYMMETRIC:
            zero_point = self.zero_point

        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            scale = self.prune_channel.apply(scale)
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: zero_point = self.prune_channel.apply(zero_point) 
        
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return quantize_per_tensor_sy(x, scale, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   quantize_per_channel_sy(x, scale, self.bitwidth, dtype=dtype)
        return quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               quantize_per_channel_assy(x, scale, zero_point, self.bitwidth, dtype=dtype)
    

class Prune_Channel:

    def __init__(
        self, 
        module,#: Sequential, 
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