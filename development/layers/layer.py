from abc import ABC, abstractmethod
from typing import Optional

import torch



class Layer(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):

        setattr(self, "_dmc", dict())
        super().__init__(*args, **kwargs)


    @abstractmethod
    def prepare_prune_channel(self):
        self.__dict__["_dmc"]["prune_channel"] = dict()
        pass

    @abstractmethod
    def apply_prune_channel(self):

        if "prune_channel" not in self.__dict__["_dmc"]:
            raise AttributeError("Layer must be prepared before applying compression")
        pass
    


    @abstractmethod
    def prepare_quantization(self, bitwidth, type):
        self.__dict__["_dmc"]["quantization"] = dict()
        self.__dict__["_dmc"]["quantization"]["bitwidth"] = bitwidth
        self.__dict__["_dmc"]["quantization"]["type"] = type


    @abstractmethod
    def apply_quantization(self):

        if "quantization" not in self.__dict__["_dmc"]:
            raise AttributeError("Layer must be prepared before applying compression")
        pass


    @abstractmethod
    def prepare_dynamic_quantization_per_tensor(
        self, 
        bitwidth,
    ):
        pass



    @abstractmethod
    def apply_dynamic_quantization_per_tensor(self):
        pass


    # @abstractmethod
    # def prepare_static_quantization_per_tensor(
    #     self,                     
    #     input_batch_real, 
    #     input_batch_quant,
    #     input_scale, 
    #     input_zero_point,
    #     bitwidth,
    # ):
    #     pass



    # @abstractmethod
    # def apply_static_quantization_per_tensor(self):
    #     pass


    # @abstractmethod
    # def set_compression_parameters(self):
    #     pass

    @abstractmethod
    def get_compression_parameters(self):
        pass


    @abstractmethod
    def get_size_in_bits(self):
        pass

    @abstractmethod
    def get_output_tensor_shape(self, input_shape):
        print("called layer")
        pass

    @abstractmethod
    def convert_to_c(self, var_name):
        pass


