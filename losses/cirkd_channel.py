import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['StudentSegChannelContrast']

class StudentSegChannelContrast(nn.Module):
    def __init__(self, channel_memory_size, channel_contrast_size,
                 contrast_kd_temperature, contrast_temperature,
                 s_size, t_size):
        super(StudentSegChannelContrast, self).__init__()
        self.base_temperature = 0.1
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        _, t_channels, t_h, t_w = t_size
        _, s_channels, s_h, s_w = s_size
        self.dim = int(t_h * t_w / (4 * 4))

        self.project_head = nn.Sequential(
            nn.Conv2d(s_channels, t_channels, 1, bias=False),
            nn.SyncBatchNorm(t_channels),
            nn.ReLU(True),
            nn.Conv2d(t_channels, t_channels, 1, bias=False)
        )

        self.channel_memory_size = channel_memory_size

        self.channel_update_freq = 256
        self.channel_contrast_size = channel_contrast_size

        self.register_buffer("teacher_channel_queue", torch.randn(self.channel_memory_size, self.dim))
        self.teacher_channel_queue = nn.functional.normalize(self.teacher_channel_queue, p=2, dim=1)
        self.register_buffer("channel_queue_ptr", torch.zeros(1, dtype=torch.long))
        

    def _sample_negative(self, Q, index):
        class_num, cache_size, feat_size = Q.shape
        contrast_size = index.size(0)
        X_ = torch.zeros((class_num * contrast_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * contrast_size, 1)).float().cuda()
        sample_ptr = 0

        
        for ii in range(class_num):
            this_q = Q[ii, index, :]
            X_[sample_ptr:sample_ptr + contrast_size, ...] = this_q
            y_[sample_ptr:sample_ptr + contrast_size, ...] = ii
            sample_ptr += contrast_size

        return X_, y_


    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    
    def _dequeue_and_enqueue(self, keys):
        channel_queue = self.teacher_channel_queue

        keys = self.concat_all_gather(keys)
        
        batch_size = keys.size(0)
        

        perm = torch.randperm(batch_size)    
        K = min(batch_size, self.channel_update_freq)
        sampled_channels = keys[perm[:K]]
        ptr = int(self.channel_queue_ptr)

        if ptr + K >= self.channel_memory_size:
            self.teacher_channel_queue[-K:, :] = nn.functional.normalize(sampled_channels, p=2, dim=1)
            self.channel_queue_ptr[0] = 0
        else:
            self.teacher_channel_queue[ptr:ptr + K, :] = nn.functional.normalize(sampled_channels, p=2, dim=1)
            self.channel_queue_ptr[0] = (ptr + K) % self.channel_memory_size


    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.contrast_kd_temperature, dim=1)
        p_t = F.softmax(t_logits/self.contrast_kd_temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature**2
        return sim_dis


    def forward(self, feat_S, feat_T):

        patch_w = 4
        patch_h = 4
        avgpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = avgpool(feat_S)
        feat_T= avgpool(feat_T)

        feat_S = self.project_head(feat_S)
        # C_S = C_T
        
        B, C_S, H, W = feat_S.size()
        B, C_T, H, W = feat_T.size()
        feat_S = feat_S.view(B, C_S, -1)
        feat_T = feat_T.view(B, C_T, -1)
        feat_S = F.normalize(feat_S, p=2, dim=-1)
        feat_T = F.normalize(feat_T, p=2, dim=-1)

        sim_dis = torch.tensor(0.).cuda()
        for i in range(B):
            for j in range(B):
                s_sim_map = torch.mm(feat_S[i], feat_S[j].transpose(0, 1))
                t_sim_map = torch.mm(feat_T[i], feat_T[j].transpose(0, 1))

                p_s = F.log_softmax(s_sim_map / self.contrast_temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.contrast_temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis += sim_dis_

        batch_channel_sim_dis = sim_dis / (B * B)
                
        feat_S = feat_S.view(B*C_S, -1)
        feat_T = feat_T.view(B*C_T, -1)

        
        channel_mse_loss = ((feat_S - feat_T) ** 2).mean()

        self._dequeue_and_enqueue(feat_T.detach().clone())

            
        channel_queue_size, feat_size = self.teacher_channel_queue.shape
        perm = torch.randperm(channel_queue_size)
        pixel_index = perm[:self.channel_contrast_size]
        t_X_channel_contrast = self.teacher_channel_queue[pixel_index, :] # (channel_contrast_size, dim)
        t_channel_logits = torch.div(torch.mm(feat_T, t_X_channel_contrast.T), self.contrast_temperature)
        s_channel_logits = torch.div(torch.mm(feat_S, t_X_channel_contrast.T), self.contrast_temperature)

        memory_channel_sim_dis = self.contrast_sim_kd(s_channel_logits, t_channel_logits.detach())
        
        return batch_channel_sim_dis, memory_channel_sim_dis, channel_mse_loss


