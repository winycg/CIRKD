CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 eval.py \
    --model psp_mobile \
    --backbone mobilenetv3_small \
    --dataset voc \
    --data /data/winycg/dataset/segmentation/VOCAug/ \
    --pretrained [your trained .pth file]

