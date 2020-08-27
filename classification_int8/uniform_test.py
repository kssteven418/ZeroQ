#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import time
import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
#from utils import *
from distill_data import *

from int_utils import *
import models.resnet as resnet
import models.resnet_original as resnet_base

input_quant_function = SymmetricQuantFunction.apply
from progress.bar import Bar

# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2',
                            'simple',
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--no_distill',
                        default=False,
                        action='store_true',
                        help='do not produce distilled dataset')
    parser.add_argument('--normal_mode',
                        default=False,
                        action='store_true',
                        help='run as a normal mode')
    parser.add_argument('--qdq',
                        default=False,
                        action='store_true',
                        help='run as a normal mode')
    parser.add_argument('--quantize',
                        default=False,
                        action='store_true',
                        help='run as a normal mode')
    args = parser.parse_args()
    return args

def test(model, test_loader):
    """
    test a model on a given dataset
    """
    total, correct = 0, 0
    bar = Bar('Testing', max=len(test_loader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total

            bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
            bar.next()
            #print(acc, correct, total)
    print('\nFinal acc: %.2f%% (%d/%d)' % (100. * acc, correct, total))
    bar.finish()
    model.train()
    return acc

def test_quantize(model, test_loader):
    bar = Bar('Testing', max=len(test_loader))
    total, correct = 0, 0
    warmup_duration = 30
    warmup = True
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if warmup:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                model(inputs)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                '''
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = correct / total
                print('batch id:', acc, correct, total)
                '''

                # finish warm-up
                if batch_idx == warmup_duration:
                    print('finishing up warmup')
                    warmup = False
                    freeze(model)
                    if args.quantize:
                        if args.qdq:
                            quantize_model(model, integer_only=False)
                        else:
                            quantize_model(model, integer_only=True)
                            model = model.cpu()
                    print('warmup done')
            else:
                # After warm-up
                if not args.quantize or args.qdq:
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()

                    # real prediction
                    real = model(inputs)
                    _, predicted = real.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = correct / total
                    #print('Real:', acc, correct, total)
                
                else:
                    # quantized prediction
                    input_q, scale_input = input_quant_function(inputs, 8)
                    q, scale_q = model((input_q, scale_input))
                    output = q.type(torch.float32) / scale_q
                    _, predicted = output.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    acc = correct / total
                    #print('Quantized:', acc, correct, total)

                bar.suffix = f'({batch_idx + 1}/{len(test_loader)}) | ETA: {bar.eta_td} | top1: {acc}'
                bar.next()

    print('\nFinal acc: %.2f%% (%d/%d)' % (100. * acc, correct, total))
    bar.finish()
    model.train()
    return acc

if __name__ == '__main__':

    test_loader = getTestData('imagenet',
                              batch_size=32,
                              path='./data/imagenet/',
                              for_inception=False)
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if not args.normal_mode:
        model_name = 'resnet18'
        model_base = ptcv_get_model(model_name, pretrained=True)
        state_dict = model_base.state_dict()
        model = resnet.resnet18()
        model.load_state_dict(state_dict, strict=False)
        if torch.cuda.is_available():
            model = model.cuda()
            model_base = model_base.cuda()
        test_quantize(model, test_loader)

    else:
        # Load pretrained model
        model = ptcv_get_model('resnet18', pretrained=True)
        #print(model, type(model))
        print('****** Full precision model loaded ******')
        #print(model)
        print(model.features)

        # Load validation data
        test_loader = getTestData('imagenet',
                                  batch_size=2,
                                  path='./data/imagenet/',
                                  for_inception=False)

        # Generate distilled data
        if torch.cuda.is_available():
            model = model.cuda()

        test(model, test_loader)

    '''
    if not args.no_distill:
        dataloader = getDistilData(
            model,
            args.dataset,
            batch_size=args.batch_size,
            for_inception=args.model.startswith('inception'))

    print('****** Data loaded ******')

    # Quantize single-precision model to 8-bit model
    quantized_model = quantize_model(model)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    if torch.cuda.is_available():
        quantized_model = quantized_model.cuda()

    # Update activation range according to distilled data
    if not args.no_distill:
        update(quantized_model, dataloader)
        freeze_model(quantized_model)

    print('****** Zero Shot Quantization Finished ******')

    # Freeze activation range during test
    quantized_model = nn.DataParallel(quantized_model)
    if torch.cuda.is_available():
        quantized_model = quantized_model.cuda()

    # Test the final quantized model
    test(quantized_model, test_loader)
    '''
