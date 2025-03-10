##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

__all__ = ['OPS', 'ResNetBasicblock', 'SearchSpaceNames']

OPS = {
  'none'         : lambda C_in, C_out, stride, affine, track_running_stats: Zero(C_in, C_out, stride),
  'avg_pool_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, 'avg', affine, track_running_stats),
  'nor_conv_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
  'nor_conv_1x1' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1), affine, track_running_stats),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}

NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

SearchSpaceNames = {'nas-bench-201': NAS_BENCH_201}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine, track_running_stats=False)
    )

  def forward(self, x):
    return self.op(x)

class ResNetBasicblock(nn.Module):

  def __init__(self, inplanes, planes, stride, affine=True):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine)
    if stride == 2:
      self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
    elif inplanes != planes:
      self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
    else:
      self.downsample = None
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def extra_repr(self):
    string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs):

    basicblock = self.conv_a(inputs)
    basicblock = self.conv_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(inputs)
    else:
      residual = inputs
    return residual + basicblock

class POOLING(nn.Module):

  def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
    super(POOLING, self).__init__()
    if C_in == C_out:
      self.preprocess = None
    else:
      self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
    if mode == 'avg'  : self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
    else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

  def forward(self, inputs):
    if self.preprocess: x = self.preprocess(inputs)
    else              : x = inputs
    return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.C_in   = C_in
    self.C_out  = C_out
    self.stride = stride
    self.is_zero = True

  def forward(self, x):
    if self.C_in == self.C_out:
      if self.stride == 1: return x.mul(0.)
      else               : return x[:,:,::self.stride,::self.stride].mul(0.)
    else:
      shape = list(x.shape)
      shape[1] = self.C_out
      zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
      return zeros

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)
