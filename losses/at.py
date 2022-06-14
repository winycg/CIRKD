import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionAT']


class CriterionAT(nn.Module):
    def __init__(self):
        super(CriterionAT, self).__init__()
        self.p = 2

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

    def forward(self, feat_S, feat_T):
        loss = (self.at(feat_S) - self.at(feat_T)).pow(2).mean()
        return loss