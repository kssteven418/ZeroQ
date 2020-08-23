from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

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


test_linear, test_conv = False, False
test_bn_folding = True
input_quant_function = SymmetricQuantFunction.apply

with torch.no_grad():
    if test_linear:

        # Test Quant_Linear
        print('=========== Quant_Linear Test ==============\n')
        linear = nn.Linear(4, 5, bias=True)
        input = torch.randn([2, 4])

        '''
        print('weight @ simple_models.py')
        print(linear.weight) 
        print()
        print('bias @ simple_models.py')
        print(linear.bias)   
        print()
        print('input @ simple_models.py')
        print(input)
        print()
        '''

        # Compute real value
        output_real = linear(input)
        #print('Output Real\n', output_real)
        #print()

        # Compute integer-only 
        ql = Quant_Linear(weight_bit=8, bias_bit=32)
        ql.set_params(linear)
        input_q, scale_input = input_quant_function(input, 8)
        output_q, scale_output = ql(input_q, scale_input)
        output = output_q.type(torch.float32) / scale_output
        #print('Output Quantized\n', output)
        #print()

        # Compute dequantize-and-floating-operation
        ql_dequant = Quant_Linear(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(linear)
        input_dq, scale_input = input_quant_function(input, 8, None, None, None, False)
        output_dq = ql_dequant(input_dq, scale_input)
        #print('Output Dequantized\n', output_dq)
        #print() 

        print('Diff (real - quant)')
        print(output_real - output)
        print()
        print('Diff (quant - dequant)')
        print(output - output_dq)
        print()
        print('============================================\n')

    if test_conv:
        # Test Quant_Conv2d
        print('=========== Quant_Conv2d Test ==============\n')
        conv = nn.Conv2d(2, 3, 2, stride=2)
        input = torch.randn([1, 2, 4, 4]) # [batch, inchannel, H, W]

        '''
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
        '''

        # Compute integer-only 
        ql = Quant_Conv2d(weight_bit=8, bias_bit=32)
        ql.set_params(conv)
        input_q, scale_input = input_quant_function(input, 8)
        output_q, scale_output = ql(input_q, scale_input)
        output = output_q.type(torch.float32) / scale_output
        #print('Output Quantized\n', output)
        #print()

        # Compute dequantize-and-floating-operation
        ql_dequant = Quant_Conv2d(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(conv)
        input_dq, scale_input = input_quant_function(input, 8, None, None, None, False)
        output_dq = ql_dequant(input_dq, scale_input)
        #print('Output Dequantized\n', output_dq)
        #print() 

        # Compute real value
        output_real = conv(input)
        #print('Output Real\n', output_real)
        #print(output_real.shape)
        #print()

        print('Diff (real - quant)')
        print(output_real - output)
        print()
        print('Diff (quant - dequant)')
        print(output - output_dq)
        print()
        print('============================================\n')

    if test_bn_folding:
        model_name = 'resnet18'
        model = ptcv_get_model(model_name, pretrained=True)
        img = torch.randn([2, 3, 200, 200])

        layer = model.features.init_block.conv
        conv = layer.conv
        bn = layer.bn
        bn.eval()

        real_conv = conv(img)
        real_bn = bn(real_conv)

        ### Test real vs. real-bn-folded
        #ql_fold = Quant_Conv2d(weight_bit=8, bias_bit=32, integer_only=False, \
        #                       full_precision_flag=True)
        #ql_fold.set_params(conv)
        #ql_fold.batchnorm_folding(bn.running_mean, bn.running_var, bn.weight, bn.bias)
        #real_folded = ql_fold(img, None)

        #print((real_folded - real_bn) / real_bn)
        #print(((real_folded - real_bn) / real_bn).max())


        ######################### 
        # Testing dquant mode
        # without bn folding
        print('dequant mode testing...')
        input_dq, scale_input = input_quant_function(img, 8, None, None, None, False)

        ql_dequant = Quant_Conv2d(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(conv)
        output_dq_conv = ql_dequant(input_dq, scale_input)

        diff_conv = real_conv - output_dq_conv
        max_real_conv = torch.abs(real_conv).max()
        max_diff_conv = torch.abs(diff_conv).max()
        print('convolution error: %f / %f = %f' % (max_diff_conv, max_real_conv, max_diff_conv / max_real_conv))

        output_dq_bn = bn(output_dq_conv)

        diff_bn = real_bn - output_dq_bn
        max_real_bn = torch.abs(real_bn).max()
        max_diff_bn = torch.abs(diff_bn).max()
        print('bn error: %f / %f = %f' % (max_diff_bn, max_real_bn, max_diff_bn / max_real_bn))

        ql_dequant.batchnorm_folding(bn.running_mean, bn.running_var, bn.weight, bn.bias)
        output_dq_fold = ql_dequant(input_dq, scale_input)

        diff_fold = real_bn - output_dq_fold
        max_diff_fold = torch.abs(diff_fold).max()
        print('fold error: %f / %f = %f' % (max_diff_fold, max_real_bn, max_diff_fold / max_real_bn))
        print()
        
        # Testing quant mode
        print('quant mode testing...')
        input_q, scale_input = input_quant_function(img, 8)

        ql = Quant_Conv2d(weight_bit=8, bias_bit=32)
        ql.set_params(conv)
        output_q, scale_output = ql(input_q, scale_input)
        output_conv = output_q.type(torch.float32) / scale_output

        diff_conv = real_conv - output_conv
        max_real_conv = torch.abs(real_conv).max()
        max_diff_conv = torch.abs(diff_conv).max()
        print('convolution error: %f / %f = %f' % (max_diff_conv, max_real_conv, max_diff_conv / max_real_conv))

        output_bn = bn(output_conv)

        diff_bn = real_bn - output_bn
        max_real_bn = torch.abs(real_bn).max()
        max_diff_bn = torch.abs(diff_bn).max()
        print('bn error: %f / %f = %f' % (max_diff_bn, max_real_bn, max_diff_bn / max_real_bn))

        ql.batchnorm_folding(bn.running_mean, bn.running_var, bn.weight, bn.bias)
        output_q, scale_output = ql(input_q, scale_input)
        output_fold = output_q.type(torch.float32) / scale_output

        diff_fold = real_bn - output_fold
        max_diff_fold = torch.abs(diff_fold).max()
        print('fold error: %f / %f = %f' % (max_diff_fold, max_real_bn, max_diff_fold / max_real_bn))
        print()
        '''
        output_real = conv(img)
        diff = output - output_real
        output_reshape = output.view(-1)
        diff_reshape = diff.view(-1)
        print(diff_reshape[diff_reshape.argmax()])
        print(output_reshape[diff_reshape.argmax()])
        '''
