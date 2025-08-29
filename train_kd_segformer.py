import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.cityscapes import CSTrainValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet

from losses import *
from utils.sagan import Discriminator
from utils.flops import cal_multi_adds, cal_param_size
from models.model_zoo import get_segmentation_model
from losses import SegCrossEntropyLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3',
                        help='model name')  
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')                      
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/',  
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--teacher-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--optimizer-type', type=str, default='sgd',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='input batch size for training (default: 8)')
    

    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--contrast-kd-temperature", type=float, default=1.0, help="similarity distribution KD temperature")
    parser.add_argument("--contrast-temperature", type=float, default=0.1, help="similarity distribution temperature")

    parser.add_argument("--lambda-kd", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-adv", type=float, default=0., help="lambda_adv")
    parser.add_argument("--lambda-d", type=float, default=0., help="lambda_d")
    parser.add_argument("--lambda-skd", type=float, default=0., help="lambda_skd")
    parser.add_argument("--lambda-cwd-fea", type=float, default=0., help="lambda_cwd feature")
    parser.add_argument("--lambda-cwd-logit", type=float, default=0., help="lambda_cwd logits")
    parser.add_argument("--lambda-ifv", type=float, default=0., help="lambda_ifv")
    parser.add_argument("--lambda-fitnet", type=float, default=0., help="lambda_fitnet")
    parser.add_argument("--lambda-at", type=float, default=0., help="lambda_attention transfer")
    parser.add_argument("--lambda-psd", type=float, default=0., help="lambda_psd")
    parser.add_argument("--lambda-csd", type=float, default=0., help="lambda_csd")

    
    # cuda setting 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local-rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--pretrained-base', type=str, default='resnet18-5c106cde.pth',
                        help='pretrained backbone')
    parser.add_argument('--pretrained', type=str, default='None',
                        help='pretrained seg model')

                        
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)


    return args
    


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(args.data, 
                                            list_path='./dataset/list/cityscapes/train.lst', 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data, 
                                        list_path='./dataset/list/cityscapes/val.lst', 
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt', max_iters=args.max_iterations*args.batch_size, 
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt', max_iters=args.max_iterations*args.batch_size,
                            ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt', max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unifind')

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.t_model = get_segmentation_model(model=args.teacher_model, 
                                            backbone=args.teacher_backbone,
                                            img_size=args.crop_size,
                                            pretrained=args.teacher_pretrained,
                                            batchnorm_layer=nn.BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.args.local_rank)

        self.s_model = get_segmentation_model(model=args.student_model, 
                                            backbone=args.student_backbone,
                                            img_size=args.crop_size,
                                            pretrained=args.student_pretrained,
                                            batchnorm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)


        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()

        self.D_model = Discriminator(preprocess_GAN_mode=1, input_channel=train_dataset.num_class, distributed=args.distributed).cuda()

        

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, 1)

        self.train_loader = data.DataLoader(dataset=train_dataset, 
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)
        

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        x = torch.randn(1,3,512,512).cuda()
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = t_y[-1].size(1)
        s_channels = s_y[-1].size(1)
        
        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_adv = CriterionAdv('hinge').to(self.device)
        self.criterion_adv_for_G = CriterionAdvForG('hinge').to(self.device)
        self.criterion_skd = CriterionStructuralKD().to(self.device)
        self.criterion_ifv = CriterionIFV(train_dataset.num_class).to(self.device)
        self.criterion_cwd = CriterionCWD(s_channels, t_channels, norm_type='channel',divergence='kl', temperature=4.).to(self.device)
        self.criterion_fitnet = CriterionFitNet(s_channels, t_channels).to(self.device)
        self.criterion_at = CriterionAT().to(self.device)
        self.criterion_dsd = CriterionDoubleSimKD().to(self.device)


        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        params_list.append(self.criterion_cwd)
        params_list.append(self.criterion_fitnet)

        if args.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(params_list.parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
        elif args.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params_list.parameters(),
                                               lr=args.lr,
                                               weight_decay=args.weight_decay)
        else:
            raise ValueError('no such optimizer')

        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            self.D_model.parameters()), 
                                            4e-4, [0.9, 0.99])

        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, 
                                                                device_ids=[args.local_rank],
                                                                output_device=args.local_rank)
            self.D_model = nn.parallel.DistributedDataParallel(self.D_model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
            
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt


    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

        
    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        self.s_model.train()
        
        for iteration,  (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            images = images.to(self.device)
            targets = targets.long().to(self.device)

            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)
            
            task_loss = self.criterion(s_outputs[0], targets)

            losses = task_loss
            
            kd_loss = torch.tensor(0.).cuda()
            adv_G_loss = torch.tensor(0.).cuda()
            adv_D_loss = torch.tensor(0.).cuda()
            skd_loss = torch.tensor(0.).cuda()
            cwd_fea_loss = torch.tensor(0.).cuda()
            cwd_logit_loss = torch.tensor(0.).cuda()
            ifv_loss = torch.tensor(0.).cuda()
            fitnet_loss = torch.tensor(0.).cuda()
            at_loss = torch.tensor(0.).cuda()
            psd_loss = torch.tensor(0.).cuda()
            csd_loss = torch.tensor(0.).cuda()
            

            adv_G_loss = self.args.lambda_adv*self.criterion_adv_for_G(self.D_model(s_outputs[0]))

            adv_D_loss = self.args.lambda_d*(self.criterion_adv(self.D_model(s_outputs[0].detach()), 
                                            self.D_model(t_outputs[0].detach())))
            
            if self.args.lambda_kd != 0.:
                kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], t_outputs[0])
            if self.args.lambda_skd != 0:
                skd_loss = self.args.lambda_skd * self.criterion_skd(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_cwd_fea != 0:
                cwd_fea_loss = self.args.lambda_cwd_fea * self.criterion_cwd(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_cwd_logit != 0:
                cwd_logit_loss = self.args.lambda_cwd_logit * self.criterion_cwd(s_outputs[0], t_outputs[0])
            if self.args.lambda_ifv != 0:
                ifv_loss = self.args.lambda_ifv * self.criterion_ifv(s_outputs[-1], t_outputs[-1], targets)
            if self.args.lambda_fitnet != 0:
                fitnet_loss = self.args.lambda_fitnet * self.criterion_fitnet(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_at != 0:
                at_loss = self.args.lambda_at * self.criterion_at(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_psd != 0. and self.args.lambda_csd != 0.:  
                feat_s_list = [s_outputs[-1], s_outputs[0]]
                feat_t_list = [t_outputs[-1], t_outputs[0]]
                psd_loss, csd_loss = self.criterion_dsd(feat_s_list, feat_t_list)
                psd_loss = self.args.lambda_psd * psd_loss
                csd_loss = self.args.lambda_csd * csd_loss

            losses = task_loss + kd_loss + adv_G_loss + \
                        skd_loss + cwd_fea_loss + cwd_logit_loss +\
                        ifv_loss + at_loss + fitnet_loss + \
                        psd_loss + csd_loss 
            D_losses = adv_D_loss

            lr = self.adjust_lr(base_lr=args.lr, iter=iteration-1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            self.D_optimizer.zero_grad()
            D_losses.backward()
            self.D_optimizer.step()

            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            kd_loss_reduced = self.reduce_mean_tensor(kd_loss)
            adv_G_loss_reduced = self.reduce_mean_tensor(adv_G_loss)
            skd_loss_reduced = self.reduce_mean_tensor(skd_loss)
            cwd_fea_loss_reduced = self.reduce_mean_tensor(cwd_fea_loss)
            cwd_logit_loss_reduced = self.reduce_mean_tensor(cwd_logit_loss)
            ifv_loss_reduced = self.reduce_mean_tensor(ifv_loss)
            at_loss_reduced = self.reduce_mean_tensor(at_loss)
            fitnet_loss_reduced = self.reduce_mean_tensor(fitnet_loss)
            psd_loss_reduced = self.reduce_mean_tensor(psd_loss)
            csd_loss_reduced = self.reduce_mean_tensor(csd_loss)
            
            
            D_losses_reduced = self.reduce_mean_tensor(D_losses)
            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}" \
                    "|| Adv_G Loss: {:.4f} || Adv_D Loss: {:.4f}" \
                    "|| skd_loss: {:.4f} || cwd_fea_loss: {:.4f} || cwd_logit_loss: {:.4f} " \
                        "|| ifv_loss: {:.4f} || at_loss: {:.4f} || fitnet_loss: {:.4f} " \
                        "|| psd_loss: {:.4f} || csd_loss: {:.4f} " \
                        "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'], 
                        task_loss_reduced.item(),
                        kd_loss_reduced.item(), 
                        adv_G_loss_reduced.item(),
                        D_losses_reduced.item(), 
                        skd_loss_reduced.item(),
                        cwd_fea_loss_reduced.item(),
                        cwd_logit_loss_reduced.item(),
                        ifv_loss_reduced.item(),
                        at_loss_reduced.item(),
                        fitnet_loss_reduced.item(),
                        psd_loss_reduced.item(),
                        csd_loss_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), 
                        eta_string))

    
            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))


    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        
        for i, (image, target, filename)  in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.long().to(self.device)

            image = torch.cat([image[:,:,:,:1024],image[:,:,:,1024:]], dim=0)
            with torch.no_grad():
                outputs = model(image)

            pred = torch.cat([outputs[0][0],outputs[0][1]], dim=-1).unsqueeze(0)
            
            B, H, W = target.size()
            pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)

            self.metric.update(pred, target)
            pixAcc, mIoU = self.metric.get()
            logger.info(str(args.local_rank) + "Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
        
        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)  # remove np.spacing(1)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            pixAcc.item() * 100, mIoU * 100))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if args.local_rank == 0:
            save_checkpoint(self.s_model, self.args, is_best)
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'kd_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
