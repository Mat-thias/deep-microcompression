import torch

from ..layers.conv import Conv2d
from ..layers.batchnorm import BatchNorm2d
from ..layers.linear import Linear
from ..layers.activation import ReLU, ReLU6
from ..layers.fused_layers import LinearReLU, Conv2dReLU, LinearReLU6, Conv2dReLU6



@torch.no_grad()
def fuse_conv2d_batchnorm2d(conv2d, batchnorm2d):
    assert isinstance(conv2d, Conv2d) and isinstance(batchnorm2d, BatchNorm2d), "conv2d has to be of Conv2d type and batchnorm2d has to be BatchNorm2d type"
    assert conv2d.out_channels == batchnorm2d.num_features, f"conv2d and batchnorm not fuseable, conv2d has {conv2d.out_channels} out_channels and batchnorm2d has {batchnorm2d.num_features} num_features, the must tbe equal"
    fused_layer = Conv2d(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        pad = conv2d.pad,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = True
    )
    fused_layer.weight.copy_(conv2d.weight * batchnorm2d.folded_weight.view(-1,1,1,1))
    if conv2d.bias is not None:
        fused_layer.bias.copy_(conv2d.bias * batchnorm2d.folded_weight + batchnorm2d.folded_bias)
    else:
        fused_layer.bias.copy_(batchnorm2d.folded_bias)
    return fused_layer


@torch.no_grad()
def fuse_linear_relu(linear, relu):
    assert isinstance(linear, Linear) and isinstance(relu, ReLU), "linear has to be of Linear type and relu has to ReLU type"
    fused_layer = LinearReLU(
        out_features = linear.out_features,
        in_features = linear.in_features,
        bias = linear.bias is not None
    )
    fused_layer.weight.copy_(linear.weight)
    if linear.bias is not None:
        fused_layer.bias.copy_(linear.bias)
    return fused_layer


@torch.no_grad()
def fuse_linear_relu6(linear, relu6):
    assert isinstance(linear, Linear) and isinstance(relu6, ReLU6), "linear has to be of Linear type and relu6 has to ReLU6 type"
    fused_layer = LinearReLU6(
        out_features = linear.out_features,
        in_features = linear.in_features,
        bias = linear.bias is not None
    )
    fused_layer.weight.copy_(linear.weight)
    if linear.bias is not None:
        fused_layer.bias.copy_(linear.bias)
    return fused_layer


@torch.no_grad()
def fuse_conv2d_relu(conv2d, relu):
    assert isinstance(conv2d, Conv2d) and isinstance(relu, ReLU), "conv2d has to be of Conv2d type and relu has to ReLU type"
    fused_layer = Conv2dReLU(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        pad = conv2d.pad,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = conv2d.bias is not None
    )
    fused_layer.weight.copy_(conv2d.weight)
    if conv2d.bias is not None and fused_layer.bias is not None:
        fused_layer.bias.copy_(conv2d.bias)
    return fused_layer




@torch.no_grad()
def fuse_conv2d_relu6(conv2d, relu6):
    assert isinstance(conv2d, Conv2d) and isinstance(relu6, ReLU6), "conv2d has to be of Conv2d type and relu6 has to ReLU6 type"
    fused_layer = Conv2dReLU6(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        pad = conv2d.pad,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = conv2d.bias is not None
    )
    fused_layer.weight.copy_(conv2d.weight)
    if conv2d.bias is not None and fused_layer.bias is not None:
        fused_layer.bias.copy_(conv2d.bias)
    return fused_layer



