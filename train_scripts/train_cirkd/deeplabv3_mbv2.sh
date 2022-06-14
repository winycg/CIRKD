CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplab_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base [your pretrained-backbone path]/mobilenetv2-imagenet.pth