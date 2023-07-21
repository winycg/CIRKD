CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
    train_baseline_segformer.py \
    --model segformer \
    --backbone MiT_B0 \
    --dataset citys \
    --batch-size 8 \
    --workers 16 \
    --lr 0.0002 \
    --optimizer-type adamw \
    --crop-size 1024 1024 \
    --max-iterations 160000 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --pretrained [your pretrained-backbone path]/mit_b0.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 \
    eval_segformer.py \
    --model segformer \
    --backbone MiT_B0 \
    --dataset citys \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --workers 16 \
    --gpu-id 0,1,2,3 \
    --pretrained [your pretrained model path]
