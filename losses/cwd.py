"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CriterionCWD']

class SpatialNorm(nn.Module):
    def __init__(self,divergence='kl'):
        if divergence =='kl':
            self.criterion = nn.KLDivLoss()
        else:
            self.criterion = nn.MSELoss()

        self.norm = nn.Softmax(dim=-1)
    
    def forward(self,pred_S,pred_T):
        norm_S = self.norm(pred_S)
        norm_T = self.norm(pred_T)

        loss = self.criterion(pred_S,pred_T)
        return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):
    def __init__(self,norm_type='none',divergence='mse',temperature=1.0):
        super(CriterionCWD, self).__init__()
        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence
        self.conv = nn.Conv2d(128, 256, kernel_size=1, bias=False)

    def forward(self,preds_S, preds_T):
        n,c,h,w = preds_S.shape
        
        if preds_S.size(1) != preds_T.size(1):
            preds_S = self.conv(preds_S)

        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()

        loss = self.criterion(norm_s,norm_t)
        
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2)