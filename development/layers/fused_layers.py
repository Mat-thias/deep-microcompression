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
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)

            if self.is_quantized:
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        input = nn.functional.linear(input, weight, bias)
        output = self.relu.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)


        return output
    


class LinearReLU6(Linear):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.relu6 = ReLU6()

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)

            if self.is_quantized:
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        input = nn.functional.linear(input, weight, bias)
        output = self.relu6.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)

        return output
    


class Conv2dReLU(Conv2d):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def forward(self, input):

        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)

            if self.is_quantized:
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)
                    
        
        input =  nn.functional.pad(input, self.pad, "constant", 0) 
        input = nn.functional.conv2d(
            input, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        output = self.relu.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)
        return output
    

class Conv2dReLU6(Conv2d):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.relu6 = ReLU6()

    def forward(self, input):

        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)

            if self.is_quantized:
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)
                    
        
        input =  nn.functional.pad(input, self.pad, "constant", 0) 
        input = nn.functional.conv2d(
            input, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        output = self.relu6.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)
        return output
    


    
