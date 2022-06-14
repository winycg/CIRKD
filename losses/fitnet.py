import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionFitNet']

class CriterionFitNet(nn.Module):
    def __init__(self):
        super(CriterionFitNet, self).__init__()
        self.conv = nn.Conv2d(128, 256, kernel_size=1, bias=False)

    def forward(self, feat_S, feat_T):
        B, C, H, W = feat_S.size()

        feat_S = self.conv(feat_S)
        '''
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        '''

        sim_err = ((feat_S - feat_T)**2)
        sim_dis = sim_err.mean()
        
        return sim_dis
