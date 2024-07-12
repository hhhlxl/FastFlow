from typing import List, Tuple

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const

class subnet_conv_ln(nn.Module):

    def __init__(self, dim_in, dim_out):
        print(dim_in, dim_out)
        super().__init__()
        dim_mid = dim_in
        self.conv1 = nn.Conv2d(dim_in, dim_mid, 3, 1, 1)
        self.ln = nn.LayerNorm(dim_mid)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim_mid, dim_out, 3, 1, 1)

    def forward(self, x):
        print('x:', x.shape)
        out = self.conv1(x)
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.conv2(out)
        print('out:', out.shape)

        return out

# 写个函数传这两个参数，但是不会写
def subnet_cbam_func(reduction, spatial_kernel):
    cbam = CBAMLayer(reduction, spatial_kernel)
    return cbam

# CBAM
class CBAMLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        # dim_in和dim_out是Fr传进来的
        super().__init__()
        reduction = 2
        channel = dim_in
        spatial_kernel = 7

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            # 这一步是为了减少隐藏层的通道数，16会不会太大了。。
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        # 增加通道数
        output = self.conv1x1(x)
        # 输出应该是64
        return output


BATCHSIZE = 8
DIMS_IN = 2
# channel height weight?
input_chw = [64, 64, 64]
input_channel = 64
clamp = 2.0

# build up basic net using SequenceINN
# Only supports a sequential series of modules (no splitting, merging, branching off).
net = Ff.SequenceINN(*input_chw)
for i in range(2):
    net.append(
        Fm.AllInOneBlock,
        subnet_constructor=CBAMLayer,
        # 这个参数用于防止指数爆炸
        affine_clamping=clamp,
        permute_soft=False,
    )

# define inputs
x = torch.randn(BATCHSIZE, 64, 64, 64)

# run forward
z, log_jac_det = net(x)

# run in reverse
x_rev, log_jac_det_rev = net(z, rev=True)