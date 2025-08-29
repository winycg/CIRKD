CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 1222 \
    train_cirkdv2.py \
    --teacher-model deeplabv3 \
    --student-model psp_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv3_small \
    --dataset coco_stuff_164k \
    --batch-size 16 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 80000 \
    --lambda-fitnet 10. \
    --lambda-minibatch-channel 1. \
    --lambda-memory-channel 0.1 \
    --lambda-channel-kd 100. \
    --data /data/winycg/dataset/segmentation/coco_stuff_164K \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/coco_stuff_164k/ \
    --save-dir-name deeplabv3_resnet101_psp_mobile_mobilenetv3_small_cirkdv2 \
    --teacher-pretrained /data/winycg/cirkd/teachers/deeplabv3_resnet101_coco_stuff_164k_best_model.pth \
    --student-pretrained-base /data/winycg/imagenet_pretrained/mobilenet_v3_small-47085aa1.pth