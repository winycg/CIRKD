CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 1379 \
    train_baseline.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset ade20k \
    --batch-size 16 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 80000 \
    --data /data/winycg/dataset/segmentation/ade20k/ADEChallengeData2016/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/ade20k/ \
    --save-dir-name deeplabv3_mobilenet_ssseg_mobilenetv3_large_ade20k_baseline \
    --pretrained-base /data/winycg/cirkd/pretrained_backbones/resnet18-imagenet.pth


