"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models.mobilenetv2 import get_mobilenet_v2

__all__ = ['get_deeplabv3_mobile']


class DeepLabV3(nn.Module):

    def __init__(self, nclass, backbone='mobilenetv2', local_rank=None, pretrained_base=True, **kwargs):
        super(DeepLabV3, self).__init__()
        
        self.pretrained = get_mobilenet_v2(pretrained=pretrained_base, local_rank=local_rank, norm_layer=kwargs['norm_layer'])
        self.head = _DeepLabHead(320, nclass, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        c4 = self.pretrained(x)

        x, x_feat_after_aspp = self.head(c4)
        return [x, x_feat_after_aspp]
        


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, norm_layer, proj_dim=256):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            norm_layer(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class _DeepLabHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(in_channels, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        out_channels = 128

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, nclass, 1)
        )
        #self.proj_head = ProjectionHead(dim_in=out_channels, norm_layer=norm_layer)

    def forward(self, x):
        x = self.aspp(x)
        x = self.block[0:4](x)
        x_feat_after_aspp = x
        x = self.block[4](x)
        return x, x_feat_after_aspp


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 128

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


def get_deeplabv3_mobile(backbone='resnet50', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=19, **kwargs):
    model = DeepLabV3(num_class, backbone=backbone, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
    return model


if __name__ == '__main__':
    model = get_deeplabv3_resnet50_voc()
    img = torch.randn(2, 3, 480, 480)
    output = model(img)
