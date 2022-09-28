
'''generate color pallete segmentation mask'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.5.0.4 --master_port 26501  \
    eval.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --dataset voc \
    --data [your dataset path]/VOCAug/ \
    --save-dir [your directory path to store segmentation mask files] \
    --save-pred \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --pretrained [your pretrained model path]

'''generate blend segmentation mask'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_addr 127.5.0.4 --master_port 26501  \
    eval.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --dataset voc \
    --data [your dataset path]/VOCAug/ \
    --save-dir [your directory path to store segmentation mask files] \
    --save-pred \
    --blend \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --pretrained [your pretrained model path]
