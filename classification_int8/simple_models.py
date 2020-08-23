from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils import *
# from distill_data import *
from int_utils import *

QTensor = namedtuple('QTensor', ['tensor', 'scale'])

class SimpleMatmul(nn.Module):
    def __init__(self, input_size):
        super(SimpleMatmul, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x

class SimpleFC(nn.Module):
    def __init__(self, input_size):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


if __name__ == '__main__':
    
    with torch.no_grad():
        input_quant_function = SymmetricQuantFunction.apply
        '''
        # Test Quant_Linear
        print('=========== Quant_Linear Test ==============\n')
        linear = nn.Linear(4, 5, bias=True)
        input = torch.randn([2, 4])

        print('weight @ simple_models.py')
        print(linear.weight) 
        print()
        print('bias @ simple_models.py')
        print(linear.bias)   
        print()
        print('input @ simple_models.py')
        print(input)
        print()

        # Compute real value
        output_real = linear(input)
        print('Output Real\n', output_real)
        print()

        # Compute integer-only 
        ql = Quant_Linear(weight_bit=8, bias_bit=32)
        print(ql)
        ql.set_params(linear)
        input_q, scale_input = input_quant_function(input, 8)
        output_q, scale_output = ql(input_q, scale_input)
        output = output_q.type(torch.float32) / scale_output
        print('Output Quantized\n', output)
        print()

        # Compute dequantize-and-floating-operation
        ql_dequant = Quant_Linear(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(linear)
        input_dq, scale_input = input_quant_function(input, 8, None, None, None, False)
        output_dq = ql_dequant(input_dq, scale_input)
        print('Output Dequantized\n', output_dq)
        print() 
        print('============================================\n')
        '''

        # Test Quant_Conv2d
        print('=========== Quant_Conv2d Test ==============\n')
        conv = nn.Conv2d(2, 3, 2, stride=2)
        input = torch.randn([1, 2, 4, 4]) # [batch, inchannel, H, W]

        print('weight @ simple_models.py')
        print(conv.weight) 
        print(conv.weight.shape)
        print()
        print('bias @ simple_models.py')
        print(conv.bias)   
        print(conv.bias.shape)
        print()
        print('input @ simple_models.py')
        print(input)
        print(input.shape)
        print()

        # Compute real value
        output_real = conv(input)
        print('Output Real\n', output_real)
        print(output_real.shape)
        print()

        # Compute dequantize-and-floating-operation
        ql_dequant = Quant_Conv2d(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(conv)
        input_dq, scale_input = input_quant_function(input, 8, None, None, None, False)
        output_dq = ql_dequant(input_dq, scale_input)
        print('Output Dequantized\n', output_dq)
        print() 
        print('============================================\n')
