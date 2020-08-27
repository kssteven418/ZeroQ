import torch
import torch.nn as nn
from int_utils import *

def quantize_model(model, integer_only=True):
    if 'full_precision_flag' in model.__dict__:
        model.full_precision_flag = False
    if 'integer_only' in model.__dict__:
        model.integer_only = integer_only

    if isinstance(model, Quant_Conv2d):
        if integer_only:
            print('bn folded')
            model.subsequent_batchnorm_folding()

    for m in model.children():
        quantize_model(m, integer_only)


def freeze(model, integer_only=True):
    if isinstance(model, Quant_Relu):
        model.fix()

    for m in model.children():
        freeze(m, integer_only)
