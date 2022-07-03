CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 train_baseline.py \
    --model deeplab_mobile \
    --backbone mobilenev2 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --pretrained-base [your pretrained-backbone path]/mobilenetv2-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch --nproc_per_node=4 eval.py \
  --model deeplab_mobile \
  --backbone mobilenev2 \
  --data [your dataset path]/cityscapes/ \
  --save-dir [your directory path to store log files] \
  --pretrained [your pretrained model path]