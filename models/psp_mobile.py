"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_models.mobilenetv2 import *


__all__ = ['get_psp_mobile']


class PSPNet(nn.Module):

    def __init__(self, nclass, backbone='resnet50', local_rank=None, pretrained_base=True, **kwargs):
        super(PSPNet, self).__init__()
        self.pretrained = get_mobilenet_v2(pretrained=pretrained_base, local_rank=local_rank, norm_layer=kwargs['norm_layer'])
        
        self.head = _PSPHead(320, nclass, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        c4 = self.pretrained(x)
        x, features = self.head(c4)

        return [x, features]


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(in_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        if in_channels == 512 or in_channels == 320:
            out_channels = 128
        elif in_channels == 2048:
            out_channels = 512
        else:
            raise "channel number error"
        self.block = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(out_channels, nclass, 1)

    def forward(self, x):
        x = self.psp(x)
        x = self.block(x)
        feature = x
        x = self.classifier(x)
        return x, feature


def get_psp_mobile(backbone='resnet50', local_rank=None,  pretrained=None,
            pretrained_base=True, num_class=19, **kwargs):

    model = PSPNet(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
    return model



if __name__ == '__main__':
    net = get_psp('citys', 'resnet18')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 512, 512)) / 1e6))

