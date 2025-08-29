CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 1366 \
    train_baseline.py \
    --model deeplabv3_mobilenet_ssseg \
    --backbone mobilenetv3_small \
    --dataset citys \
    --batch-size 8 \
    --lr 0.01 \
    --max-iterations 80000 \
    --data /data/winycg/dataset/segmentation/cityscape/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/cityscapes/ \
    --save-dir-name deeplabv3_mobilenet_ssseg_mobilenetv3_small_citys_baseline \
    --pretrained-base /data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth
