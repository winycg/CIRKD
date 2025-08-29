CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 1366 \
    train_baseline.py \
    --model psp_mobile \
    --backbone mobilevit_xx_small \
    --dataset citys \
    --batch-size 8 \
    --lr 0.0009 \
    --max-iterations 80000 \
    --optimizer-type adamw \
    --weight-decay 0.01 \
    --data /data/winycg/dataset/segmentation/cityscape/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/cityscapes/ \
    --save-dir-name psp_mobile_mobilevit_xx_small_citys_baseline \
    --pretrained-base /data/winycg/imagenet_pretrained/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth
