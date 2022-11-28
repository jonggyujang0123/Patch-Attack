import torch.nn as nn
import torch
import math
from torch.nn import functional as F



class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias_init=0, lr_mul=1, activation=None, use_bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=False)

        if activation == 'fused_lrelu':
            self.activation = activation
        else:
            self.activation = None
            
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.bias_init = bias_init

    def forward(self, input):
        bias = self.bias * self.lr_mul + self.bias_init
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(input, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

