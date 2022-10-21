CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 train_baseline.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset camvid \
    --crop-size 360 360 \
    --data [your dataset path]/CamVid/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0 \
  python -m torch.distributed.launch --nproc_per_node=1 \
  eval.py \
  --model deeplabv3 \
  --backbone resnet18 \
  --dataset camvid \
  --data [your dataset path]/CamVid/ \
  --save-dir [your directory path to store checkpoint files] \
  --pretrained [your pretrained model path]