from typing import Optional

import torch
from torch import nn

from .layer import Layer


class BatchNorm2d(nn.BatchNorm2d, Layer):


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    
    def forward(self, input):

        input = super().forward(input)
        return input
    

    @torch.no_grad()
    def set_compression_parameters(self):

        if getattr(self, "pruned", False):
            self.set_prune_parameters()

        return
    

    @torch.no_grad()
    def get_compression_parameters(self):
        
        weight = self.weight
        bias = self.bias
        running_mean = self.running_mean
        running_var = self.running_var

        if getattr(self, "pruned", False):
            weight, bias, running_mean, running_var = self.get_prune_parameters()

        return weight, bias, running_mean, running_var
    
    
    @torch.no_grad
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Optional[torch.Tensor], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        super().prune_channel(sparsity, keep_prev_channel_index, is_output_layer, metric)

        setattr(self, "keep_prev_channel_index", keep_prev_channel_index)

        return keep_prev_channel_index
    
    @torch.no_grad
    def set_prune_parameters(self):
        
        mask = torch.zeros_like(self.weight)
        mask[getattr(self, "keep_prev_channel_index")] = 1

        self.weight.mul_(mask)
        self.bias.mul_(mask)
        self.running_mean.mul_(mask)
        self.running_var.mul_(mask)

        return 
    
    @torch.no_grad
    def get_prune_parameters(self):
        
        weight = torch.index_select(self.weight, 0, getattr(self, "keep_prev_channel_index"))
        bias = torch.index_select(self.bias, 0, getattr(self, "keep_prev_channel_index"))
        running_mean = torch.index_select(self.running_mean, 0, getattr(self, "keep_prev_channel_index"))
        running_var = torch.index_select(self.running_var, 0, getattr(self, "keep_prev_channel_index"))

        return weight, bias, running_mean, running_var
        
    