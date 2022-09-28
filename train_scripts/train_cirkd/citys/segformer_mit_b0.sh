CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8  --master_addr 127.5.0.4 --master_port 26501 \
    train_cirkd_segformer.py \
    --teacher-model segformer \
    --student-model segformer \
    --teacher-backbone MiT_B4 \
    --student-backbone MiT_B0 \
    --dataset citys \
    --data --data [your dataset path]/cityscapes/ \
    --batch-size 8 \
    --workers 16 \
    --crop-size 1024 1024 \
    --optimizer-type adamw \
    --pixel-memory-size 2000 \
    --region-contrast-size 1024 \
    --pixel-contrast-size  4096 \
    --kd-temperature 1 \
    --lambda-kd 1.0 \
    --contrast-kd-temperature 1. \
    --lambda-kd 1. \
    --lambda-minibatch-pixel 1. \
    --lambda-memory-pixel 0.1 \
    --lambda-memory-region 0.1 \
    --lr 0.0002 \
    --max-iterations 160000 \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --teacher-pretrained [your teacher weights path]/segformer_MiT_B4_citys_best_model.pth \
    --student-pretrained [your pretrained-backbone path]/mit_b0.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    eval_segformer.py \
    --model segformer \
    --backbone MiT_B0 \
    --dataset citys \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --workers 16 \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --pretrained [your pretrained model path]

