CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 3333 \
    train_cirkdv2.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3_mobilenet_ssseg \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv3_large \
    --dataset ade20k \
    --batch-size 16 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 80000 \
    --lambda-fitnet 10. \
    --lambda-minibatch-channel 1. \
    --lambda-memory-channel 0.1 \
    --lambda-channel-kd 100. \
    --data /data/winycg/dataset/segmentation/ade20k/ADEChallengeData2016/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/ade20k/ \
    --save-dir-name deeplabv3_resnet101_deeplabv3_mobilenet_ssseg_mobilenetv3_large_cirkdv2 \
    --teacher-pretrained /data/winycg/cirkd/teachers/deeplabv3_resnet101_ade20k_best_model.pth \
    --student-pretrained-base /data/winycg/imagenet_pretrained/mobilenet_v3_large-bc2c3fd3.pth