CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_kd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-kd 1.0 \
    --lambda-d 0.1 \
    --lambda-adv 0.001 \
    --lambda-cwd-fea 50. \
    --lambda-cwd-logit 3. \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base [your pretrained-backbone path]/resnet18-imagenet.pth



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_kd.py \
    --teacher-model deeplabv3 \
    --student-model deeplab_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --lambda-kd 1.0 \
    --lambda-d 0.1 \
    --lambda-adv 0.001 \
    --lambda-cwd-fea 50. \
    --lambda-cwd-logit 3. \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --teacher-pretrained [your teacher weights path]/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base [your pretrained-backbone path]/mobilenetv2-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.launch --nproc_per_node=2  --master_addr 127.5.0.4 --master_port 26501 \
    train_kd_segformer.py \
    --teacher-model segformer \
    --student-model segformer \
    --teacher-backbone MiT_B4 \
    --student-backbone MiT_B0 \
    --dataset citys \
    --data [your dataset path]/cityscapes/ \
    --batch-size 8 \
    --workers 16 \
    --crop-size 1024 1024 \
    --optimizer-type adamw \
    --lambda-kd 1.0 \
    --lambda-d 0.1 \
    --lambda-adv 0.001 \
    --lambda-cwd-fea 5. \
    --lambda-cwd-logit 0.3 \
    --lr 0.0002 \
    --max-iterations 160000 \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --teacher-pretrained [your teacher weights path]/segformer_MiT_B4_citys_best_model.pth \
    --student-pretrained [your pretrained-backbone path]/mit_b0.pth