import sys
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .quant_utils import *

class Quant_Module(nn.Module):
    def __init__(self, weight_bit=8, bias_bit=32, downcast=True,
                 full_precision_flag=False, integer_only=True):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or frozen
        """
        super(Quant_Module, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.integer_only = integer_only
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.weight_bit_function = SymmetricQuantFunction.apply
        self.downcast = downcast

    def __repr__(self):
        s = super(Quant_Module, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, full_precision_flag={})".format(
                 self.weight_bit, self.bias_bit, self.full_precision_flag)
        return s


class Quant_Conv2d(Quant_Module):
    def __init__(self, weight_bit=8, bias_bit=32, downcast=False,
                 full_precision_flag=False, integer_only=True):
        super(Quant_Conv2d, self).__init__(weight_bit, bias_bit, downcast,
                                           full_precision_flag, integer_only)

    def set_params(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        # For now, just store parameters as float32
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def batchnorm_folding(self, mean, var, weight, bias):
        mean, var, weight, bias = mean.data, var.data, weight.data, bias.data
        if self.bias is None:
            self.bias = Parameter(torch.zeros([self.out_channels]))
        bias_data = self.bias.data.clone()
        weight_data = self.weight.data.clone()

        bias_data = bias_data + bias - \
                    mean * weight / torch.sqrt(1e-8 + var)
        weight_data = weight_data * weight.view(-1, 1, 1, 1) / \
                      torch.sqrt(1e-8 + var.view(-1, 1, 1, 1))
        self.bias.copy_(bias_data)
        self.weight.copy_(weight_data)

    def forward(self, x_q, scale_x):
        w = self.weight

        if self.full_precision_flag:
            return F.conv2d(x_q, w, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)

        # o, i, h, w -> o, iwh
        w_transform = w.data.contiguous().view(self.out_channels, -1)
        w_min = w_transform.min(dim=1).values
        w_max = w_transform.max(dim=1).values

        w_q, scale_w = self.weight_bit_function(self.weight, self.weight_bit, 
            w_min, w_max, None, self.integer_only, 'Conv2d_w')
        scale_out = scale_w * scale_x
        if self.bias is None:
            b_q = None
        else:
            b_q, _ = self.weight_bit_function(self.bias, self.bias_bit, 
                    None, None, scale_out, self.integer_only, 'Conv2d_b')
        # dequantization-and-floating-operation path for comparison and debugging purpose
        if not self.integer_only:
            assert w_q.dtype == torch.float32
            assert x_q.dtype == torch.float32
            assert b_q is None or b_q.dtype == torch.float32
            return F.conv2d(x_q, w_q, b_q, self.stride, self.padding,
                            self.dilation, self.groups)

        # integer-only-operation-path
        # cast x_q and w_q from int8 to int32
        x_q = x_q.type(torch.int32)
        w_q = w_q.type(torch.int32)

        assert b_q is None or b_q.dtype == torch.int32

        out_q = F.conv2d(x_q, w_q, b_q, self.stride, self.padding,
                         self.dilation, self.groups)

        # scale factor for conv2d output is out_channel-wise
        scale_out = scale_out.view(1, -1, 1, 1)
        if self.downcast:
            return downcast_function(out_q, scale_out)
        return out_q, scale_out


class Quant_Linear(Quant_Module):
    def __init__(self, weight_bit=8, bias_bit=32, downcast=True,
                 full_precision_flag=False, integer_only=True):
        super(Quant_Linear, self).__init__(weight_bit, bias_bit, downcast,
                                           full_precision_flag, integer_only)

    def set_params(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # For now, just store parameters as float32
        self.weight = Parameter(linear.weight.data.clone())
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
       
    def forward(self, x_q, scale_x):
        w = self.weight # float32

        if self.full_precision_flag:
            return F.linear(x_q, weight=w, bias=self.bias)

        w_transform = w.data.detach()
        w_min = w_transform.min(dim=1).values
        w_max = w_transform.max(dim=1).values

        # this will produce dequantized float32 value
        w_q, scale_w = self.weight_bit_function(self.weight, self.weight_bit, 
            w_min, w_max, None, self.integer_only, 'Linear_w')
        scale_out = scale_w * scale_x
        if self.bias is None:
            b_q = None
        else:
            b_q, _ = self.weight_bit_function(self.bias, self.bias_bit, 
                    None, None, scale_out, self.integer_only, 'Linear_b')

        # dequantization-and-floating-operation path for comparison and debugging purpose
        if not self.integer_only:
            assert w_q.dtype == torch.float32
            assert x_q.dtype == torch.float32
            assert b_q is None or b_q.dtype == torch.float32
            return F.linear(x_q, weight=w_q, bias=b_q)

        # integer-only-operation-path
        # cast x_q and w_q from int8 to int32
        x_q = x_q.type(torch.int32)
        w_q = w_q.type(torch.int32)

        assert b_q is None or b_q.dtype == torch.int32
        
        out_q = F.linear(x_q, weight=w_q, bias=b_q)

        # scale factor for matmul output is row-wise
        scale_out = scale_out.view(1, -1)
        if self.downcast:
            return downcast_function(out_q, scale_out)
        return out_q, scale_out
