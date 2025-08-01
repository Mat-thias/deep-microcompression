from abc import ABC, abstractmethod


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



