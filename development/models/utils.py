import torch
from torch import nn
from .sequential import Sequential
from ..layers import *


def convert_from_sequential_torch_to_dmc(torch_model: nn.Sequential) -> Sequential:

    assert isinstance(torch_model, (nn.Sequential or Sequential)), "Model must be a torch sequential model."

    dmc_model = Sequential()
    for module in torch_model:
        dmc_model = dmc_model + convert_from_layer_torch_nn_to_dmc(module)
    return dmc_model


def convert_from_layer_torch_nn_to_dmc(module: nn.Module):
    
    @torch.no_grad()
    def copy_tensor(tensor_source, tensor_destination):
        tensor_destination.copy_(tensor_source)

    if isinstance(module, nn.Conv2d):
        pad = [module.padding[0]]*2 + [module.padding[1]]*2
        layer = Conv2d(
            in_channels=module.in_channels, out_channels=module.out_channels, 
            kernel_size=module.kernel_size, stride=module.stride, pad=pad, 
            bias=module.bias is not None
        )
        copy_tensor(module.weight, layer.weight)
        if layer.bias is not None:
            copy_tensor(module.bias, layer.bias)

    elif isinstance(module, nn.Linear):
        layer = Linear(
            in_features=module.in_features, out_features=module.out_features, 
            bias=module.bias is not None
        )
        copy_tensor(module.weight, layer.weight)
        if layer.bias is not None:
            copy_tensor(module.bias, layer.bias)


    elif isinstance(module, nn.ReLU):
        layer = ReLU(inplace=module.inplace) 

    elif isinstance(module, nn.ReLU6):
        layer = ReLU6(inplace=module.inplace) 


    elif isinstance(module, nn.Flatten):
        layer = Flatten(start_dim=module.start_dim, end_dim=module.end_dim) 
        
    elif isinstance(module, nn.MaxPool2d):
        layer = MaxPool2d(
            module.kernel_size, module.stride,
            module.padding, module.dilation,
            ceil_mode=module.ceil_mode, return_indices=module.return_indices
        )

    elif isinstance(module, nn.AvgPool2d):
        layer = AvgPool2d(
            module.kernel_size, module.stride,
            module.padding, module.dilation,
            ceil_mode=module.ceil_mode, return_indices=module.return_indices
        )
        
    elif isinstance(module, nn.BatchNorm2d):
        
        layer = BatchNorm2d(num_features=module.num_features, eps=module.eps, momentum=module.momentum, affine=module.affine,  track_running_stats=module.track_running_stats)
        if module.affine:
            copy_tensor(module.weight, layer.weight)
            copy_tensor(module.bias, layer.bias)
        if module.track_running_stats:
            copy_tensor(module.running_mean, layer.running_mean)
            copy_tensor(module.running_var, layer.running_var)

    elif isinstance(module, nn.Dropout):
        return ReLU()
    else:
        raise RuntimeError(f"module of type {type(module)} does not have a dmc equivalent yet.")


    return layer
                
        