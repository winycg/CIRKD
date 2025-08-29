CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12365 \
    test.py \
    --model upernet_lite \
    --backbone resnet18 \
    --dataset citys \
    --data /data/winycg/dataset/segmentation/cityscape/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/visualization \
    --save-pred \
    --pretrained [your trained .pth file]


