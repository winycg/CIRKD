CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_baseline.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --dataset ade20k \
    --batch-size 16 \
    --crop-size 512 512 \
    --lr 0.02 \
    --data [your dataset path]/ade20k/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --pretrained-base [your pretrained-backbone path]/resnet101-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 eval.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --dataset ade20k \
    --data [your dataset path]/ade20k/ \
    --save-dir [your directory path to store checkpoint files] \
    --pretrained [your pretrained model path]