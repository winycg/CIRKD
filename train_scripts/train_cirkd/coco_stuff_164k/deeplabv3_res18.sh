CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8  --master_addr 127.5.0.4 --master_port 26501 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset coco_stuff_164k \
    --data [your dataset path]/coco_stuff_164k/ \
    --batch-size 16 \
    --workers 8 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 80000 \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path]/deeplabv3_resnet101_coco_stuff_164k_best_model.pth \
    --student-pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --master_addr 127.5.0.9 --master_port 2601 --nproc_per_node=8 \
    eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset coco_stuff_164k \
    --data [your dataset path]/coco_stuff_164k/ \
    --save-dir [your directory path to store checkpoint files] \
    --pretrained [your pretrained model path]
