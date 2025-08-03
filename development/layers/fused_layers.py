
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
    

class Conv2dReLU(Conv2d):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def forward(self, input):
        input = super().forward(input)
        input = self.relu.forward(input)

        return input
    