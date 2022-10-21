CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 train_baseline.py \
    --model psp \
    --backbone resnet18 \
    --dataset voc \
    --crop-size 512 512 \
    --data [your dataset path]/VOCAug/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 eval.py \
    --model psp \
    --backbone resnet18 \
    --dataset voc \
    --data [your dataset path]/VOCAug/ \
    --save-dir [your directory path to store checkpoint files] \
    --pretrained [your pretrained model path]