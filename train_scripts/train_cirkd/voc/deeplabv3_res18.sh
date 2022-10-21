CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_cirkd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --dataset voc \
    --crop-size 512 512 \
    --data [your dataset path]/VOCAug/ \
    --lambda-memory-pixel 0.01 \
	--lambda-memory-region 0.01 \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path] \
    --student-pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.launch --nproc_per_node=3 eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset voc \
    --data [your dataset path]/VOCAug/ \
    --save-dir [your directory path to store checkpoint files] \
    --pretrained [your pretrained model path]