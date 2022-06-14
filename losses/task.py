import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SegCrossEntropyLoss']

class SegCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, **kwargs):
        super(SegCrossEntropyLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        B, H, W = targets.size()
        inputs = F.interpolate(inputs, (H, W), mode='bilinear', align_corners=True)
        return self.task_loss(inputs, targets)