import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['StudentSegContrast']

class StudentSegContrast(nn.Module):
    def __init__(self, num_classes, pixel_memory_size, region_memory_size, region_contrast_size, pixel_contrast_size, 
                 contrast_kd_temperature, contrast_temperature, ignore_label):
        super(StudentSegContrast, self).__init__()
        self.base_temperature = 0.1
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        self.dim = 256
        self.ignore_label = ignore_label
        self.n_view = 32


        self.project_head = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.SyncBatchNorm(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 1, bias=False)
        )

        self.num_classes = num_classes
        self.region_memory_size = region_memory_size
        self.pixel_memory_size = pixel_memory_size
        self.pixel_update_freq = 16
        self.pixel_contrast_size = pixel_contrast_size
        self.region_contrast_size = region_contrast_size


        self.register_buffer("teacher_segment_queue", torch.randn(self.num_classes, self.region_memory_size, self.dim))
        self.teacher_segment_queue = nn.functional.normalize(self.teacher_segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("teacher_pixel_queue", torch.randn(self.num_classes, self.pixel_memory_size, self.dim))
        self.teacher_pixel_queue = nn.functional.normalize(self.teacher_pixel_queue, p=2, dim=2)
        self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))


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

    
    def _dequeue_and_enqueue(self, keys, labels):
        segment_queue = self.teacher_segment_queue
        pixel_queue = self.teacher_pixel_queue

        keys = self.concat_all_gather(keys)
        labels = self.concat_all_gather(labels)
        
        batch_size, feat_dim, H, W = keys.size()

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x != self.ignore_label]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.region_memory_size

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)    
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.pixel_memory_size:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + K) % self.pixel_memory_size


    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.contrast_kd_temperature, dim=1)
        p_t = F.softmax(t_logits/self.contrast_kd_temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature**2
        return sim_dis


    def forward(self, s_feats, t_feats, labels=None, predict=None):
        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = self.project_head(s_feats)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (s_feats.shape[2], s_feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == s_feats.shape[-1], '{} {}'.format(labels.shape, s_feats.shape)

        ori_s_fea = s_feats
        ori_t_fea = t_feats
        ori_labels = labels

        batch_size = s_feats.shape[0]

        labels = labels.contiguous().view(-1)
        predict = predict.contiguous().view(batch_size, -1)

        idxs = (labels != self.ignore_label)
        s_feats = s_feats.permute(0, 2, 3, 1)
        s_feats = s_feats.contiguous().view(-1, s_feats.shape[-1])
        s_feats = s_feats[idxs, :]

        t_feats = t_feats.permute(0, 2, 3, 1)
        t_feats = t_feats.contiguous().view(-1, t_feats.shape[-1])
        t_feats = t_feats[idxs, :]

        self._dequeue_and_enqueue(ori_t_fea.detach().clone(), ori_labels.detach().clone())

        if idxs.sum() == 0: # just a trick to 
            return 0. * (s_feats**2).mean(), 0. * (s_feats**2).mean(), 0. * (s_feats**2).mean()
            
        class_num, pixel_queue_size, feat_size = self.teacher_pixel_queue.shape
        perm = torch.randperm(pixel_queue_size)
        pixel_index = perm[:self.pixel_contrast_size]
        t_X_pixel_contrast, t_y_pixel_contrast = self._sample_negative(self.teacher_pixel_queue, pixel_index)


        t_pixel_logits = torch.div(torch.mm(t_feats, t_X_pixel_contrast.T), self.contrast_temperature)
        s_pixel_logits = torch.div(torch.mm(s_feats, t_X_pixel_contrast.T), self.contrast_temperature)


        class_num, region_queue_size, feat_size = self.teacher_segment_queue.shape
        perm = torch.randperm(region_queue_size)
        region_index = perm[:self.region_contrast_size]
        t_X_region_contrast, _ = self._sample_negative(self.teacher_segment_queue, region_index)

        
        t_region_logits = torch.div(torch.mm(t_feats, t_X_region_contrast.T), self.contrast_temperature)
        s_region_logits = torch.div(torch.mm(s_feats, t_X_region_contrast.T), self.contrast_temperature)

        pixel_sim_dis = self.contrast_sim_kd(s_pixel_logits, t_pixel_logits.detach())
        region_sim_dis = self.contrast_sim_kd(s_region_logits, t_region_logits.detach())
        
        return pixel_sim_dis, region_sim_dis


