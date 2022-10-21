CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model psp \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset camvid \
    --crop-size 360 360 \
    --data [your dataset path]/CamVid/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth

CUDA_VISIBLE_DEVICES=0 \
  python -m torch.distributed.launch --nproc_per_node=1 \
  eval.py \
  --model psp \
  --backbone resnet18 \
  --dataset camvid \
  --data [your dataset path]/CamVid/ \
  --save-dir [your directory path to store checkpoint files] \
  --pretrained [your pretrained model path]
