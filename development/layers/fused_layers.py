import torch
from torch import nn

from .layer import Layer
from .linear import Linear
from .conv import Conv2d
from .activation import ReLU, ReLU6


class LinearReLU(Linear):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def forward(self, input):
        input = super().forward(input)
        input = self.relu.forward(input)

        return input
    

    
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
                # if scheme == QuantizationScheme.DYNAMIC:
                #     setattr(self, "bias_quantize", Quantize(
                #         self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize]
                #     ))
                if scheme == QuantizationScheme.STATIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, STATIC_BIAS_BITWDHT, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize, self.input_quantize]
                    ))
            else:
                # if scheme == QuantizationScheme.DYNAMIC:
                #     setattr(self, "bias_quantize", Quantize(
                #         self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize], prune_channel=self.bias_prune_channel
                #     ))
                if scheme == QuantizationScheme.STATIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, STATIC_BIAS_BITWDHT, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize, self.input_quantize], prune_channel=self.bias_prune_channel
                    ))

        # calibration
        if scheme == QuantizationScheme.DYNAMIC:
            self.weight_quantize.update_parameters(self.weight) 
            # if self.bias is not None:
            #     self.bias_quantize.update_parameters(self.bias)
 


    

class Conv2dReLU(Conv2d):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def forward(self, input):
        input = super().forward(input)
        input = self.relu.forward(input)

        return input
    