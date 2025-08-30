from typing import Optional
from math import prod

import torch

from ..layers.layer import Layer



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

        # self.keep_current_channel_index.to(self.module.device)
        # self.keep_prev_channel_index.to(self.module.device)


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