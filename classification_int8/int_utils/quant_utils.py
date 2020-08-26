import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import torch.nn as nn

def adjust_shape(target, source):
    """
    view the source Tensor to make it has the same dimension as the target Tensor.
    """
    # covolution weights and activations
    if len(source.shape) == 4:  
        return target.view(-1, 1, 1, 1)
    # linear weights
    elif len(source.shape) == 2:
        return target.view(-1, 1)
    # bias
    elif len(source.shape) == 1 or len(source.shape) == 0:
        return target
    raise NotImplementedError


def clamp_per_feature(input, min, max):
    min = adjust_shape(min, input)
    max = adjust_shape(max, input)
    return torch.max(torch.min(input, max), min)


def linear_quantize(input, scale, qtype=torch.int8):
    if qtype == torch.int8:
        k = 8
    if qtype == torch.int32:
        k = 32
    n = 2 ** (k-1)
    min, max = -n, n-1
    scale_reshape = adjust_shape(scale, input) 
    qtensor = torch.round(scale_reshape * input).clamp(min, max).type(qtype)
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
    

def linear_quantization_params(num_bits, qrange, is_symmetric):
    """
    Compute the scaling factor with the given quantization range [-qrange, qrange].
    """
    n = 2**num_bits - 1 # range length in the quantization domain
    # range length in the real domain 
    # range is [-qrange, qrange] if symmetric, [0, qrange] if asymmetric
    qrange_len = 2 * qrange if is_symmetric else qrange
    scale = n / torch.clamp((qrange_len), min=1e-6)
    return scale

def downcast_function(x, scale, output_dtype=torch.int8):
    """
    x: quantized integer input
    scale: scale factor of the input
    output_dtype: desired dtype of the output
    """
    #print('before downcast:', x, scale)
    if x.dtype == torch.int32:
        input_bit = 32
    else:
        raise NotImplementedError('int32 is only supported for downcast input.')
    if output_dtype == torch.int8:
        output_bit = 8
    else:
        raise NotImplementedError('int8 is only supported for downcast output.')

    '''
    scale_cast = 2**(input_bit - output_bit)
    assert scale_cast >= 1
    scale_out = scale * scale_cast
    output = (x // scale_cast).type(output_dtype)
    print('after downcast:', x // scale_cast, scale_out)
    return  output, scale_out
    '''
    x_min, x_max = x.min().type(torch.float32), x.max().type(torch.float32)
    qrange = torch.max(torch.abs(x_min), torch.abs(x_max))
    scale_cast = linear_quantization_params(output_bit, qrange, is_symmetric=True)
    output_cast, scale_cast = linear_quantize(x, scale_cast, output_dtype)
    #print('after downcast:', output_cast, scale_cast * scale)
    return output_cast, scale_cast * scale

def requantization_function(x, scale, target_scale, shift=16):
    print()
    print('scale')
    print(scale)
    print('target_scale')
    print(target_scale)
    print()
    n = 2**shift
    multiplier = (target_scale / scale * n).type(x.dtype)
    #print('MULTIPLIER')
    #print(multiplier)
    x = x * multiplier
    x = x // n
    return x

class Addition(nn.Module):
    def __init__(self, full_precision_flag=True, integer_only=True):
       super(Addition, self).__init__() 
       self.full_precision_flag = full_precision_flag
       self.integer_only = integer_only

    def __repr__(self):
         return "{0}(full_precision_flag={1}, integer_only={2})".format(
                 self.__class__.__name__, self.full_precision_flag, self.integer_only)

    def forward(self, x, y):
        if not self.integer_only or self.full_precision_flag:
            return x + y
        else:
            assert isinstance(x, tuple) and isinstance(y, tuple)
            x, scale_x = x
            y, scale_y = y
            y = y.type(x.dtype)
            print()
            print('Addition y')
            print(y)
            print(y.dtype)
            print()
            y_rescaled = requantization_function(y, scale_y, scale_x, shift=2)
            print()
            print('Addition y_rescaled')
            print(y_rescaled)
            print(y_rescaled.dtype)
            print()
            return x + y_rescaled, scale_x

class SymmetricQuantFunction(Function):

    @staticmethod
    def forward(self, x, k, x_min=None, x_max=None, scale=None, 
                integer_only=True, name=None):
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
            raise NotImplementedError("only 8-bit and 32-bit quantizations are supported")

        # This path is for quantizing bias w.r.t the given scaling factor
        if scale is not None:
            qrange = (2**(k-1) - 1) / scale
        # This path is for quantizing weights or activations where scaling factor is not given
        else:
            if x_min is None or x_max is None or \
                    (sum(x_min == x_max) == 1 and x_min.numel() == 1):
                x_min, x_max = x.min(), x.max()
            qrange = torch.max(torch.abs(x_min), torch.abs(x_max))
            scale = linear_quantization_params(k, qrange, is_symmetric=True)

        qtensor, scale = linear_quantize(clamp_per_feature(x, -qrange, qrange), scale, qtype)
        if integer_only:
            assert qtensor.dtype == qtype
            return qtensor, scale
        else:
            dqtensor = linear_dequantize(qtensor, scale)
            assert dqtensor.dtype == torch.float32
            return dqtensor

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
        

if __name__ == '__main__':

    # clamp per feature test
    w = torch.Tensor([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
    print(w)
    min = torch.Tensor([-2, -1, 0])
    max = torch.Tensor([0, 1, 2])
    w_clamp = clamp_per_feature(w, min, max)
    print(w_clamp)


    # Test linear quantization and dequantization
    qfunction = SymmetricQuantizationFunction.apply
    x = torch.randn([4, 5])
    qtensor, scale = qfunction(x, 8)
    print(qtensor)
    print(scale)

    tensor = linear_dequantize(qtensor, scale)
    print(tensor)
    print(x)
