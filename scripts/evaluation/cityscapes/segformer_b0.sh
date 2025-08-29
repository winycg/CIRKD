CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    test_segformer.py \
    --model segformer \
    --backbone MiT_B0 \
    --dataset citys \
    --data /data/winycg/dataset/segmentation/cityscape/ \
    --save-dir /data/winycg/checkpoints/cirkd_checkpoints/visualization \
    --save-pred \
    --pretrained [your trained model .pth file]