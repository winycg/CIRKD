from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionDoubleSimKD']


class CriterionPSD(nn.Module):
    def __init__(self):
        super(CriterionPSD, self).__init__()
        self.p = 2

    def attention_preprocess(self, f_list):
        outs = []
        for f in f_list:
            outs.append(F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1)))
        return outs

    def residual_attention(self, f_list):
        ra_list = []
        for n in range(len(f_list)-1):
            for m in range(n+1, len(f_list)):
                ra = F.normalize(f_list[m]-f_list[n])
                ra_list.append(ra)
        return ra_list

    def forward(self, feat_S_list, feat_T_list):
        feat_S_list = self.attention_preprocess(feat_S_list)
        feat_T_list = self.attention_preprocess(feat_T_list)

        ra_S_list = self.residual_attention(feat_S_list)
        ra_T_list = self.residual_attention(feat_T_list)

        K = len(ra_S_list)
        psd_loss = torch.tensor(0.).cuda()
        for k in range(K):
            psd_loss += (F.normalize(ra_S_list[k])-F.normalize(ra_T_list[k])).pow(2).mean()
        
        psd_loss = psd_loss / K
        return psd_loss


class CriterionCSD(nn.Module):
    def __init__(self):
        super(CriterionCSD, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        self.tau = 4.

    def pair_wise_sim_map(self, fea):
        B, C, H, W = fea.size()
        fea = fea.reshape(B, C, -1)
        fea = F.softmax(fea/self.tau, dim=-1)
        fea_T = fea.transpose(1,2)
        sim_map = torch.bmm(fea, fea_T)
        return sim_map

    def forward(self, feat_S, feat_T):
        feat_S = self.pooling(feat_S)
        feat_T= self.pooling(feat_T)

        S_sim_map = self.pair_wise_sim_map(feat_S)
        T_sim_map = self.pair_wise_sim_map(feat_T)

        sim_err = ((S_sim_map - T_sim_map)**2)
        sim_dis = sim_err.mean()

        return sim_dis


class CriterionDoubleSimKD(nn.Module):
    def __init__(self):
        super(CriterionDoubleSimKD, self).__init__()
        self.criterionPSD = CriterionPSD()
        self.criterionCSD = CriterionCSD()

    def forward(self, feat_S_list, feat_T_list):
        psd_loss = self.criterionPSD(feat_S_list, feat_T_list)
        csd_loss = self.criterionCSD(feat_S_list[-1], feat_T_list[-1])
        
        return psd_loss, csd_loss