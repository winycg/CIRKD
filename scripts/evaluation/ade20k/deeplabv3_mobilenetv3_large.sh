CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 eval.py \
    --model deeplabv3_mobilenet_ssseg \
    --backbone mobilenetv3_large \
    --dataset ade20k \
    --data /data/winycg/dataset/segmentation/ade20k/ADEChallengeData2016/ \
    --pretrained [your trained .pth file]




