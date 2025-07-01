from abc import ABC, abstractmethod
from typing import Optional

import torch



class Layer:

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Optional[torch.Tensor], 
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        setattr(self, "pruned", True)
        print("KJHLLLLLLLLLLLLLL")
        pass

    @abstractmethod
    def set_compression_parameters(self):
        pass

    @abstractmethod
    def get_compression_parameters(self):
        pass

    @abstractmethod
    def set_prune_parameters(self):
        pass

    @abstractmethod
    def get_prune_parameters(self):
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


