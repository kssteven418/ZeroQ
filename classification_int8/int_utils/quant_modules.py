import sys
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .quant_utils import *

class Quant_Linear(nn.Module):
    def __init__(self, weight_bit=8, bias_bit=32, 
                 full_precision_flag=False, integer_only=True):
        """
        weight: bit-setting for weight
        full_precision_flag: full precision or not
        running_stat: determines whether the activation range is updated or frozen
        """
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.integer_only = integer_only
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.weight_bit_function = SymmetricQuantFunction.apply

    def __repr__(self):
        s = super(Quant_Linear, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, full_precision_flag={})".format(
                 self.weight_bit, self.bias_bit, self.full_precision_flag)
        return s

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
        w_transform = w.data.detach()
        w_min = w_transform.min(dim=1).values
        w_max = w_transform.max(dim=1).values

        if self.full_precision_flag:
            raise NotImplementedError
            #return F.linear(x, weight=w, bias=self.bias)

        # dequantization-and-floating-operation path
        # simply for comparison and debugging purpose
        if not self.integer_only:
            # this will produce dequantized float32 value
            w_q, scale_w = self.weight_bit_function(self.weight, self.weight_bit, 
                w_min, w_max, None, self.integer_only, 'Linear_w')
            scale_out = scale_w * scale_x
            assert w_q.dtype == torch.float32
            assert x_q.dtype == torch.float32

            if self.bias is None:
                b_q = None
            else:
                b_q, _ = self.weight_bit_function(self.bias, self.bias_bit, 
                        None, None, scale_out, self.integer_only, 'Linear_b')
                assert b_q.dtype == torch.float32
            return F.linear(x_q, weight=w_q, bias=b_q)

        # integer-only-operation-path
        w_q, scale_w = self.weight_bit_function(self.weight, self.weight_bit, 
                w_min, w_max, None, self.integer_only, 'Linear_w')
        scale_out = scale_w * scale_x

        # cast x_q and w_q from int8 to int32
        x_q = x_q.type(torch.int32)
        w_q = w_q.type(torch.int32)

        if self.bias is None:
            b_q = None
        else:
            b_q, _ = self.weight_bit_function(self.bias, self.bias_bit, 
                    None, None, scale_out, self.integer_only, 'Linear_b')
            assert b_q.dtype == torch.int32
        
        print('Quantized values in Quant_Linear class')
        print('Xq:', x_q, x_q.shape)
        print('Wq:', w_q, w_q.shape)
        if b_q is not None:
            print('bq:', b_q, b_q.shape)
        print()
        out_q = F.linear(x_q, weight=w_q, bias=b_q)

        return out_q, scale_out.view(1, -1)
