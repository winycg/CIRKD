import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['GCN']

class GCN(nn.Module):
    def __init__(self, plane, norm_layer):
        super(GCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1, bias=False)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1, bias=False)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1, bias=False)


        self.conv1 = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1, bias=False),
                                   norm_layer(plane),
                                   nn.ReLU())

        self.out_conv = nn.Sequential(nn.Conv2d(plane * 2, plane, kernel_size=1, bias=False),
                                      norm_layer(plane),
                                      nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()

        node_q = node_q.view(b, c, -1).permute(0, 2, 1)
        node_k = node_k.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)

        sim_map = torch.bmm(node_q * (c**-.5), node_k)
        normalized_sim_map = self.softmax(sim_map)

        gcn_feats = torch.bmm(normalized_sim_map, node_v)
        gcn_feats = gcn_feats.transpose(1, 2).contiguous().view(b, c, h, w)
        gcn_feats = self.conv1(gcn_feats)

        out_feats = torch.cat([x, gcn_feats], dim=1)
        out_feats = self.out_conv(out_feats)
        return out_feats
