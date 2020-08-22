from collections import namedtuple
import math
import numpy as np
from torch.autograd import Function, Variable
import torch

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
    return qtensor, scale
    

def linear_dequantize(qinput, scale):
    dtype = scale.dtype

    # covolution weights and activations
    if len(qinput.shape) == 4:  
        scale = scale.view(-1, 1, 1, 1)
    # linear weights
    elif len(qinput.shape) == 2:
        scale = scale.view(-1, 1)
    return qinput.type(dtype) / scale
    

def symmetric_linear_quantization_params(num_bits, qrange):
    """
    Compute the scaling factor with the given quantization range [-qrange, qrange].
    """
    n = 2**num_bits - 1
    scale = n / torch.clamp((2 * qrange), min=1e-8)
    return scale


class SymmetricQuantizationFunction(Function):

    @staticmethod
    def forward(self, x, k, x_min=None, x_max=None, scale=None, name=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range 
        x_max: upper bound for quantization range 
        """
        if k == 8:
            qtype = torch.int8
        elif k == 32:
            qtype = torch.int32
        else:
            raise NotImplementedError

        # This case is for quantizing bias w.r.t the given scaling factor
        if scale is not None:
            qrange = (2**(k-1) - 1) / scale
        # This case is for quantizing weights or activations where scaling factor is not given
        else:
            if x_min is None or x_max is None or \
                    (sum(x_min == x_max) == 1 and x_min.numel() == 1):
                x_min, x_max = x.min(), x.max()
            qrange = torch.max(-x_min, x_max)
            scale = symmetric_linear_quantization_params(k, qrange)
            qtensor, scale = linear_quantize(torch.clamp(x, -qrange, qrange), scale, qtype)
        #return (torch.autograd.Variable(qtensor.tensor, qtensor.scale))
        return qtensor, scale

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
        

if __name__ == '__main__':
    qfunction = SymmetricQuantizationFunction.apply
    x = torch.randn([4, 5])
    qtensor, scale = qfunction(x, 8)
    print(qtensor)
    print(scale)

    tensor = linear_dequantize(qtensor, scale)
    print(tensor)
    print(x)
