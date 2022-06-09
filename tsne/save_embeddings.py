from __future__ import print_function

import os
import sys
import argparse
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from models.model_zoo import get_segmentation_model

from utils.score import SegmentationMetric
from utils.distributed import *
from utils.logger import setup_logger

from dataset.cityscapes import CSValSet


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn32s)')  
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--data', type=str, default='./dataset/VOCdevkit',  
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default='./dataset/list/cityscapes/val.lst',  
                        help='dataset directory')
    parser.add_argument('--base-size', default=[2048, 1024], type=int, nargs='+', help='base image size: [width, height]')
    parser.add_argument('--crop-size', type=int, default=[512, 512], nargs='+',
                        help='crop image size: [width, height]')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0') 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default='psp_resnet18_citys_best_model.pth',
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='../runs/logs/',
                        help='Directory for saving predictions')
    parser.add_argument('--save-pred', action='store_true', default=False,
                    help='save predictions')
    args = parser.parse_args()

    if args.backbone.startswith('resnet'):
        args.aux = True
    else:
        args.aux = False

    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)

        # dataset and dataloader
        
        val_dataset = CSValSet(args.data, os.path.join(os.getcwd(), '../dataset/list/cityscapes/val.lst'), crop_size=(1024, 2048))

        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model,
                                            backbone=args.backbone, 
                                            local_rank=args.local_rank,
                                            pretrained=args.pretrained, 
                                            pretrained_base='None',
                                            aux=args.aux, 
                                            norm_layer=BatchNorm2d,
                                            num_class=val_dataset.num_class).to(self.device)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))

        overall_embeddings = []
        overall_labels = []
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.float().to(self.device)

            print('progress: {}/{}'.format(i, len(self.val_loader)))
            if i == 50:
                break

            with torch.no_grad():
                outputs = model(image)

            embeddings = outputs[-1]
            B, C, H, W= embeddings.size()
            embeddings = embeddings.permute(0, 2, 3, 1)
            embeddings = embeddings.contiguous().view(-1, embeddings.shape[-1])

            labels = target
            labels = F.interpolate(labels.unsqueeze(1), (H, W), mode='nearest')
            labels = labels.permute(0, 2, 3, 1)
            labels = labels.contiguous().view(-1, 1)
            
            index_1 =(~(labels == -1)).squeeze(-1)
            embeddings = embeddings[index_1]
            labels = labels[index_1]


            overall_embeddings.append(embeddings)
            overall_labels.append(labels)


        if self.args.local_rank == 0:
            overall_embeddings = torch.cat(overall_embeddings, dim=0)
            overall_labels = torch.cat(overall_labels, dim=0)

            print('overall_embeddings', overall_embeddings.size())
            print('overall_labels', overall_labels.size())

            overall_embeddings = overall_embeddings.cpu().numpy()
            overall_labels = overall_labels.cpu().numpy()
            
            import numpy as np
            np.save('seg_embeddings.npy', overall_embeddings)
            np.save('seg_labels.npy', overall_labels)
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    if args.save_pred:
        outdir = '{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        outdir = os.path.join(args.save_dir, outdir)
        if (args.distributed and args.local_rank == 0) or args.distributed is False:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args, num_gpus)
    evaluator.eval()
    torch.cuda.empty_cache()
