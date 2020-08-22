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

import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *


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
                            'resnet20_cifar10', 'sqnxt23_w2'
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')

    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/imagenet/',
                              for_inception=args.model.startswith('inception'))

    # Generate distilled data
    if torch.cuda.is_available():
        model = model.cuda()

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
