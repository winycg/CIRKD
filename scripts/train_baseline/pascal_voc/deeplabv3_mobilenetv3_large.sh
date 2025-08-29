CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 1378 \
    train_baseline.py \
    --model deeplabv3_mobilenet_ssseg \
    --backbone mobilenetv3_large \
    --dataset voc \
    --batch-size 16 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 80000 \
    --data /data/winycg/dataset/segmentation/VOCAug/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/voc/ \
    --save-dir-name deeplabv3_mobilenet_ssseg_mobilenetv3_large_voc_baseline \
    --pretrained-base /data/winycg/imagenet_pretrained/mobilenet_v3_large-bc2c3fd3.pth
