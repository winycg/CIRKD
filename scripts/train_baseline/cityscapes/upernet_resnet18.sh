CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 1366 \
    train_baseline.py \
    --model upernet_lite \
    --backbone resnet18 \
    --dataset citys \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --data /data/winycg/dataset/segmentation/cityscape/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/cityscapes/ \
    --save-dir-name upernet_lite_resnet18_citys_baseline \
    --pretrained-base /data/winycg/cirkd/pretrained_backbones/resnet18-imagenet.pth


