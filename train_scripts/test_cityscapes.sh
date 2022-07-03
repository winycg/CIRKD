CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch --nproc_per_node=4 test.py \
  --model deeplabv3 \
  --backbone resnet101 \
  --data [your dataset path]/cityscapes/ \
  --save-dir [your directory path to store resulting images] \
  --save-pred \
  --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth