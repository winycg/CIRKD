import math
from statistics import mode
from tkinter.tix import MAIN
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import os

__all__ = ['MiT_B0', 'MiT_B1', 'MiT_B2', 'get_segformer']

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        #img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
    
    
class LinearMLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    
    
class Segformer(nn.Module):
    def __init__(
        self, 
        pretrained=None,
        img_size=1024, 
        patch_size=4, 
        in_chans=3, 
        num_classes=150, 
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4], 
        qkv_bias=True, 
        qk_scale=None, 
        drop_rate=0.,
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        batchnorm_layer=nn.BatchNorm2d,
        depths=[3, 6, 40, 3], 
        sr_ratios=[8, 4, 2, 1],
        decoder_dim = 256
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, 
            patch_size=7, 
            stride=4, 
            in_chans=in_chans,
            embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=(img_size[0] // 4, img_size[1] // 4),
            patch_size=3, 
            stride=2, 
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=(img_size[0] // 8, img_size[1] // 8),
            patch_size=3, 
            stride=2, 
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=(img_size[0] // 16, img_size[1] // 16), 
            patch_size=3, 
            stride=2, 
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3]
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        
        
        # segmentation head
        self.linear_c4 = LinearMLP(input_dim=embed_dims[3], embed_dim=decoder_dim)
        self.linear_c3 = LinearMLP(input_dim=embed_dims[2], embed_dim=decoder_dim)
        self.linear_c2 = LinearMLP(input_dim=embed_dims[1], embed_dim=decoder_dim)
        self.linear_c1 = LinearMLP(input_dim=embed_dims[0], embed_dim=decoder_dim)
        self.linear_fuse = nn.Sequential(*[nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
                                           batchnorm_layer(decoder_dim),
                                           nn.ReLU()])
        self.dropout = nn.Dropout2d(drop_rate)
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            this_dir = os.getcwd()
            pretrained =  os.path.join(this_dir, pretrained)
            old_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            self.load_state_dict(model_dict)
        

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        
        c1, c2, c3, c4 = x 
        
         ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        h_out, w_out = c1.size()[2], c1.size()[3]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size = c1.size()[2:], mode = 'bilinear', align_corners = False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim = 1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
    
        return [x, _c]


def MiT_B0(pretrained, img_size, num_classes, batchnorm_layer):
    model = Segformer(
        pretrained=pretrained,
        img_size=img_size,
        patch_size=4, 
        num_classes=num_classes,
        embed_dims=[32, 64, 160, 256], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        batchnorm_layer=batchnorm_layer,
        depths=[2, 2, 2, 2], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1,
        decoder_dim = 256
    )
    return model

def MiT_B1(pretrained, img_size, num_classes, batchnorm_layer):
    model = Segformer(
        pretrained=pretrained,
        img_size=img_size,
        patch_size=4, 
        num_classes=num_classes,
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        batchnorm_layer=batchnorm_layer,
        depths=[2, 2, 2, 2], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1,
        decoder_dim = 256
    )
    return model


def MiT_B2(pretrained, img_size, num_classes, batchnorm_layer):
    model = Segformer(
        pretrained=pretrained,
        img_size=img_size,
        patch_size=4, 
        num_classes=num_classes,
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        batchnorm_layer=batchnorm_layer,
        depths=[3, 4, 6, 3], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1,
        decoder_dim = 768
    )
    return model


def MiT_B3(pretrained, img_size, num_classes, batchnorm_layer):
    model = Segformer(
        pretrained=pretrained,
        img_size=img_size,
        patch_size=4, 
        num_classes=num_classes,
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        batchnorm_layer=batchnorm_layer,
        depths=[3, 4, 18, 3], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1,
        decoder_dim = 768
    )
    return model


def MiT_B4(pretrained, img_size, num_classes, batchnorm_layer):
    model = Segformer(
        pretrained=pretrained,
        img_size=img_size,
        patch_size=4, 
        num_classes=num_classes,
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        batchnorm_layer=batchnorm_layer,
        depths=[3, 8, 27, 3], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1,
        decoder_dim = 768
    )
    return model


def MiT_B5(pretrained, img_size, num_classes, batchnorm_layer):
    model = Segformer(
        pretrained=pretrained,
        img_size=img_size,
        patch_size=4, 
        num_classes=num_classes,
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        batchnorm_layer=batchnorm_layer,
        depths=[3, 6, 40, 3], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1,
        decoder_dim = 768
    )
    return model


def get_segformer(backbone, pretrained, img_size, num_class, batchnorm_layer):
    if backbone == 'MiT_B0':
        return MiT_B0(pretrained, img_size, num_class, batchnorm_layer)
    elif backbone == 'MiT_B1':
        return MiT_B1(pretrained, img_size, num_class, batchnorm_layer)
    elif backbone == 'MiT_B2':
        return MiT_B2(pretrained, img_size, num_class, batchnorm_layer)
    elif backbone == 'MiT_B3':
        return MiT_B3(pretrained, img_size, num_class, batchnorm_layer)
    elif backbone == 'MiT_B4':
        return MiT_B4(pretrained, img_size, num_class, batchnorm_layer)
    else:
        raise ValueError('no such backbone')

if __name__ == '__main__':
    import sys
    import os.path
    this_dir = os.getcwd()
    sys.path.insert(0, this_dir + '/..')

    from utils import cal_param_size, get_flops

    model = MiT_B0(pretrained='../pretrained_backbones/mit_b0.pth',
                   img_size=(1024, 1024),
                   num_classes=19,
                   batchnorm_layer=nn.BatchNorm2d).cuda()

    x = torch.ones(1, 3, 1024, 1024).cuda()
    y = model(x)
    print('y[0]', y[0].size())
    print('y[1]', y[1].size())
    print('y[2]', y[2].size())
    exit()
    details = get_flops(model, x.size(), net_type='transformer')
    params = cal_param_size(model)
    print(details*1e-9)
    print(params*1e-6)
