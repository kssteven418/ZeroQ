from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

from int_utils import *

test_linear, test_conv = True, True
test_bn_folding = True
test_relu = False
test_e2e = False
input_quant_function = SymmetricQuantFunction.apply


with torch.no_grad():

    if test_linear:

        # Test Quant_Linear
        print('=========== Quant_Linear Test ==============\n')
        linear = nn.Linear(3, 4, bias=True)
        input = torch.randn([2, 3])

        # Compute real value
        output_real = linear(input)
        print('Output Real\n', output_real)
        print()

        # Compute integer-only 
        ql = Quant_Linear(weight_bit=8, bias_bit=32)
        ql.set_params(linear)
        input_q, scale_input = input_quant_function(input, 8)
        output_q, scale_output = ql((input_q, scale_input))
        output = output_q.type(torch.float32) / scale_output
        #print('Output Quantized\n', output)
        #print()

        # Compute dequantize-and-floating-operation
        ql_dequant = Quant_Linear(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(linear)
        input_dq = input_quant_function(input, 8, None, None, None, False)
        output_dq = ql_dequant(input_dq)
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

        # Compute real value
        output_real = conv(input)
        #print('Output Real\n', output_real)
        #print(output_real.shape)
        #print()

        # Compute integer-only 
        ql = Quant_Conv2d(weight_bit=8, bias_bit=32)
        ql.set_params(conv)
        input_q, scale_input = input_quant_function(input, 8)
        output_q, scale_output = ql((input_q, scale_input))
        output = output_q.type(torch.float32) / scale_output
        #print('Output Quantized\n', output)
        #print()

        # Compute dequantize-and-floating-operation
        ql_dequant = Quant_Conv2d(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(conv)
        input_dq = input_quant_function(input, 8, None, None, None, False)
        output_dq = ql_dequant(input_dq)
        #print('Output Dequantized\n', output_dq)
        #print() 


        print('Diff (real - quant)')
        print(output_real - output)
        print()
        print('Diff (quant - dequant)')
        print(output - output_dq)
        print()
        print('============================================\n')


    if test_relu:
        # Test Quant_Linear
        print('=========== Quant_Relu Test ==============\n')
        print('=========== Simple Example ===============\n')
        qact = Quant_Relu(8, full_precision_flag=True)

        # full precision and unfixed
        for i in range(10):
            input = torch.randn([2, 4])
            qact(input, None)

        qact.fix()
        qact.full_precision_flag = False

        print(qact.x_max)
        print(qact.scale_out)

        input = torch.randn([2, 4])
        real = F.relu(input)
        print('real')
        print(real)
        print()

        input_q, scale_input_q = input_quant_function(input, 8)
        input_q = input_q.type(torch.int32)

        output_q, scale_q = qact(input_q, scale_input_q)
        print('output_q')
        print(output_q, scale_q)
        output = output_q.type(torch.float32) / scale_q
        print('output')
        print(output)
        print()


        print('=========== COmplex Example ===============\n')
        model_name = 'resnet18'
        model = ptcv_get_model(model_name, pretrained=True)

        layer = model.features.init_block.conv
        shape = [2, 3, 200, 200]
        layer = model.features.stage4.unit1.body.conv1
        shape = [2, 256, 200, 200]

        qact = Quant_Relu(8, full_precision_flag=True)
        conv = layer.conv
        bn = layer.bn
        bn.eval()

        # full precision and unfixed
        for i in range(10):
            input = torch.randn(shape)
            output_bn = bn(conv(input))
            qact(output_bn, None)

        qact.fix()
        qact.full_precision_flag = False

        print(qact.x_max)
        print(qact.scale_out)

        input = torch.randn(shape)
        output_bn = bn(conv(input))
        real = F.relu(output_bn)
        max_real_relu = torch.abs(real).max()

        output_bn_q, scale_output_bn_q = input_quant_function(output_bn, 8)
        output_bn_q = output_bn_q.type(torch.int32)

        output_q, scale_q = qact(output_bn_q, scale_output_bn_q)
        output = output_q.type(torch.float32) / scale_q

        diff_relu = real - output
        max_diff_relu = torch.abs(diff_relu).max()
        print('relu error: %f / %f = %f' % (max_diff_relu, max_real_relu, 
                                            max_diff_relu / max_real_relu))
        print()


    if test_e2e:
        print('=========== Conv-BN-Relu E2E Test ==============\n')
        model_name = 'resnet18'
        model = ptcv_get_model(model_name, pretrained=True)
        print(type(model))

        layer = model.features.stage4.unit1.body.conv1
        shape = [2, 256, 200, 200]
        layer = model.features.init_block.conv
        shape = [2, 3, 200, 200]

        conv = layer.conv
        bn = layer.bn
        bn.eval()

        ql = Quant_Conv2d(weight_bit=8, bias_bit=32)
        ql.set_params(conv)
        ql.batchnorm_folding(bn.running_mean, bn.running_var, bn.weight, bn.bias)
        qact = Quant_Relu(8, full_precision_flag=True)

        # full precision and unfixed
        for i in range(1):
            input = torch.randn(shape)
            output_bn = bn(conv(input))
            qact(output_bn, None)

        qact.fix()
        qact.full_precision_flag = False

        print(qact.x_max)
        print(qact.scale_out)

        input = torch.randn(shape)
        real_relu = layer(input)
        #print(real_relu)
        max_real_relu = torch.abs(real_relu).max()

        input_q, scale_input = input_quant_function(input, 8)

        output_q_fold, scale_fold = ql(input_q, scale_input)
        output_q_relu, scale_relu = qact(output_q_fold, scale_fold)
        output_relu = output_q_relu.type(torch.float32) / scale_relu

        #print(output_relu)

        diff_relu = real_relu - output_relu
        print(diff_relu)
        print(real_relu)
        
        max_diff_relu = torch.abs(diff_relu).max()
        print('relu error: %f / %f = %f' % (max_diff_relu, max_real_relu, 
                                            max_diff_relu / max_real_relu))
        print()

    if test_bn_folding:
        print('==============Test BN Folding===============\n')
        print()
        model_name = 'resnet18'
        model = ptcv_get_model(model_name, pretrained=True)

        layer = model.features.stage4.unit1.body.conv1
        img = torch.randn([2, 256, 200, 200])
        layer = model.features.init_block.conv
        img = torch.randn([2, 3, 200, 200])

        conv = layer.conv
        bn = layer.bn
        bn.eval()

        real_conv = conv(img)
        real_bn = bn(real_conv)

        ######################### 
        # Testing dquant mode
        # without bn folding
        print('dequant mode testing...')
        input_dq = input_quant_function(img, 8, None, None, None, False)

        ql_dequant = Quant_Conv2d(weight_bit=8, bias_bit=32, integer_only=False)
        ql_dequant.set_params(conv)
        output_dq_conv = ql_dequant(input_dq)

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
        output_dq_fold = ql_dequant(input_dq)

        diff_fold = real_bn - output_dq_fold
        max_diff_fold = torch.abs(diff_fold).max()
        print('fold error: %f / %f = %f' % (max_diff_fold, max_real_bn, max_diff_fold / max_real_bn))
        print()
        
        # Testing quant mode
        print('quant mode testing...')
        input_q, scale_input = input_quant_function(img, 8)

        ql = Quant_Conv2d(weight_bit=8, bias_bit=32)
        ql.set_params(conv)
        output_q, scale_output = ql((input_q, scale_input))
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
        output_q, scale_output = ql((input_q, scale_input))
        output_fold = output_q.type(torch.float32) / scale_output

        diff_fold = real_bn - output_fold
        max_diff_fold = torch.abs(diff_fold).max()
        print('fold error: %f / %f = %f' % (max_diff_fold, max_real_bn, max_diff_fold / max_real_bn))
        print()
