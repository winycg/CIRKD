import torch
import torch.nn as nn
from .segbase import SegBaseModel

__all__ = ['get_upernet', 'get_upernet_lite']


def conv3x3_bn_relu(in_planes, out_planes, stride=1, norm_layer=nn.BatchNorm2d):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True),
            )


class UPerNetHead(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256,
                 norm_layer=nn.BatchNorm2d):
        super(UPerNetHead, self).__init__()

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1, norm_layer)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 2),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last[0](fusion_out)
        feature = x
        x = self.conv_last[1](x)
        return x, feature

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)

class UPerNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', fpn_dim=512, aux=False, local_rank=None, pretrained_base=True, **kwargs):
        super(UPerNet, self).__init__(nclass, aux, backbone, local_rank, pretrained_base=pretrained_base, **kwargs)
        self.aux = aux
        if backbone.startswith('resnet18') or backbone.startswith('resnet34'):
            fpn_inplanes = [64, 128, 256, 512]
            in_channels = 512
        else:
            fpn_inplanes = [256, 512, 1024, 2048]
            in_channels = 2048

        self.head = UPerNetHead(
                num_class=nclass,
                fc_dim=in_channels,
                fpn_inplanes=fpn_inplanes,
                fpn_dim=fpn_dim)
        if self.aux:
            self.auxlayer = _FCNHead(in_channels // 2, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])
    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        conv_out = [c1, c2, c3, c4]
        x, x_feat_after_aspp = self.head(conv_out)

        if self.aux:
            auxout = self.auxlayer(c3)
        return [x, auxout, x_feat_after_aspp]



def get_upernet(backbone='resnet18_nodilated', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=19, **kwargs):

    model = UPerNet(num_class, backbone=backbone, fpn_dim=512, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
    return model

def get_upernet_lite(backbone='resnet18_nodilated', local_rank=None, pretrained=None, 
                  pretrained_base=True, num_class=19, **kwargs):

    model = UPerNet(num_class, backbone=backbone, fpn_dim=128, local_rank=local_rank, pretrained_base=pretrained_base, **kwargs)
    if pretrained != 'None':
        if local_rank is not None:
            device = torch.device(local_rank)
            model.load_state_dict(torch.load(pretrained, map_location=device))
    return model

