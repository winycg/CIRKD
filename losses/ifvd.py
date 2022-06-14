import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionIFV']


class CriterionIFV(nn.Module):
    def __init__(self, classes):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes

    def forward(self, feat_S, feat_T, target):
        preds_T = feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return loss