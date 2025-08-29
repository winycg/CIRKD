CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12397 \
    train_cirkdv2.py \
    --teacher-model deeplabv3 \
    --student-model psp_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilevit_xx_small \
    --batch-size 8 \
    --lr 0.0009 \
    --max-iterations 80000 \
    --optimizer-type adamw \
    --weight-decay 0.01 \
    --lambda-fitnet 1. \
    --lambda-minibatch-channel 1. \
    --lambda-memory-channel 0.1 \
    --lambda-channel-kd 100. \
    --data /data/winycg/dataset/segmentation/cityscape/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/cityscapes/ \
    --save-dir-name deeplabv3_resnet101_psp_mobilevit_xx_small_cirkdv2 \
    --teacher-pretrained /data/winycg/cirkd/teachers/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base /data/winycg/imagenet_pretrained/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth