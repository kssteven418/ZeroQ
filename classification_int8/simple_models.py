from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from distill_data import *
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
    batch_size = 4
    input_size = 8
    model = SimpleMatmul(input_size)

    input = torch.randn([batch_size, input_size])
    print(model(input))

    quantized_model = quantize_model(model) 
    print(quantized_model)
