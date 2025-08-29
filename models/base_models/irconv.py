'''
Function:
    Implementation of InvertedResidual and InvertedResidualV3
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation import BuildActivation
from .normalization import BuildNormalization
import math 


__all__ = ['AdptivePaddingConv2d', 'InvertedResidual', 'InvertedResidualV3']


'''makedivisible'''
def makedivisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None: min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value: new_value += divisor
    return new_value


class SqueezeExcitationConv2d(nn.Module):
    def __init__(self, channels, ratio=16, act_cfgs=None, makedivisible_args={'divisor': 8}):
        super(SqueezeExcitationConv2d, self).__init__()
        assert act_cfgs is not None, 'argument act_cfgs should be given'
        assert len(act_cfgs) == 2, 'length of act_cfgs should be equal to 2'
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        act_cfg = act_cfgs[0]
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Conv2d(channels, makedivisible(channels//ratio, **makedivisible_args), kernel_size=1, stride=1, padding=0))
        self.conv1.add_module('activation', BuildActivation(act_cfg))
        act_cfg = act_cfgs[1]
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv', nn.Conv2d(makedivisible(channels//ratio, **makedivisible_args), channels, kernel_size=1, stride=1, padding=0))
        self.conv2.add_module('activation', BuildActivation(act_cfg))
    '''forward'''
    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class AdptivePaddingConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm_cfg=None, act_cfg=None):
        super(AdptivePaddingConv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        if norm_cfg is not None: 
            self.norm = BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg)
        if act_cfg is not None: 
            self.activation = BuildActivation(act_cfg)
    '''forward'''
    def forward(self, x):
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = (max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0))
        pad_w = (max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0))
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if hasattr(self, 'norm'): output = self.norm(output)
        if hasattr(self, 'activation'): output = self.activation(output)
        return output


'''InvertedResidual'''
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, dilation=1, norm_cfg=None, act_cfg=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s' % stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layer = nn.Sequential()
            layer.add_module('conv', nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            if norm_cfg is not None:
                layer.add_module('bn', BuildNormalization(placeholder=hidden_dim, norm_cfg=norm_cfg))
            if act_cfg is not None:
                layer.add_module('activation', BuildActivation(act_cfg))
            layers.append(layer)
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_dim, bias=False))
        if norm_cfg is not None:
            layer.add_module('bn', BuildNormalization(placeholder=hidden_dim, norm_cfg=norm_cfg))
        if act_cfg is not None:
            layer.add_module('activation', BuildActivation(act_cfg))
        layers.extend([layer])
        layer = nn.Sequential()
        layer.add_module('conv', nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if norm_cfg is not None:
            layer.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
        layers.extend([layer])
        self.conv = nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


'''InvertedResidualV3'''
class InvertedResidualV3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size=3, stride=1, se_cfg=None, with_expand_conv=True, norm_cfg=None, act_cfg=None):
        super(InvertedResidualV3, self).__init__()
        assert stride in [1, 2], 'stride must in [1, 2], but received %s' % stride
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        self.with_expand_conv = with_expand_conv
        if not self.with_expand_conv: assert mid_channels == in_channels
        if self.with_expand_conv:
            self.expand_conv = nn.Sequential()
            self.expand_conv.add_module('conv', nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False))
            if norm_cfg is not None:
                self.expand_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.expand_conv.add_module('activation', BuildActivation(act_cfg))
        self.depthwise_conv = nn.Sequential()
        if stride == 2:
            self.depthwise_conv.add_module('conv', AdptivePaddingConv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
            if norm_cfg is not None:
                self.depthwise_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.depthwise_conv.add_module('activation', BuildActivation(act_cfg))
        else:
            self.depthwise_conv.add_module('conv', nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False))
            if norm_cfg is not None:
                self.depthwise_conv.add_module('bn', BuildNormalization(placeholder=mid_channels, norm_cfg=norm_cfg))
            if act_cfg is not None:
                self.depthwise_conv.add_module('activation', BuildActivation(act_cfg))
        if se_cfg is not None:
            self.se = SqueezeExcitationConv2d(**se_cfg)
        self.linear_conv = nn.Sequential()
        self.linear_conv.add_module('conv', nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if norm_cfg is not None:
            self.linear_conv.add_module('bn', BuildNormalization(placeholder=out_channels, norm_cfg=norm_cfg))
    '''forward'''
    def forward(self, x):
        out = x
        if self.with_expand_conv: out = self.expand_conv(out)
        out = self.depthwise_conv(out)
        if hasattr(self, 'se'): out = self.se(out)
        out = self.linear_conv(out)
        if self.with_res_shortcut:
            return x + out
        return out