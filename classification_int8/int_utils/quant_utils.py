from collections import namedtuple
import math
import numpy as np
from torch.autograd import Function, Variable
import torch

QTensor = namedtuple('QTensor', ['tensor', 'scale'])

def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, qtype=torch.int8):

    # covolution weights and activations
    if len(input.shape) == 4:  
        scale = scale.view(-1, 1, 1, 1)
    # linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
    
    qtensor = (scale * input).type(qtype)
    return QTensor(qtensor, scale)
    

def linear_dequantize(quantized_input):
    input = quantized_input.tensor
    scale = quantized_input.scale
    dtype = scale.dtype

    # covolution weights and activations
    if len(input.shape) == 4:  
        scale = scale.view(-1, 1, 1, 1)
    # linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
    return input.type(dtype) / scale
    

def symmetric_linear_quantization_params(num_bits, 
                                         saturation_min, 
                                         saturation_max):
    """
    Compute the scaling factor with the given quantization range [saturation_min, saturation_max].
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    return scale


class SymmetricQuantizationFunction(Function):

    @staticmethod
    def forward(self, x, k, x_min=None, x_max=None, name=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range 
        x_max: upper bound for quantization range 
        """
        if x_min is None or x_max is None or \
                (sum(x_min == x_max) == 1 and x_min.numel() == 1):
            x_min, x_max = x.min(), x.max()
        scale = symmetric_linear_quantization_params(k, x_min, x_max)

if __name__ == '__main__':
    x = torch.randn([4, 5])
    k, qtype = 8, torch.int8
    print(x)
    x_min, x_max = x.min(), x.max()
    scale = symmetric_linear_quantization_params(k, x_min, x_max)
    print(scale)
    qtensor = linear_quantize(x, scale, qtype)
    print(qtensor)
    tensor = linear_dequantize(qtensor)
    print(tensor)
