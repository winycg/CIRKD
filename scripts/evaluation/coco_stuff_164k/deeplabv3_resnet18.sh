CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset coco_stuff_164k \
    --data /data/winycg/dataset/segmentation/coco_stuff_164K/ \
    --pretrained [your trained .pth file]


