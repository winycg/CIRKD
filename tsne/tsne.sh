CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 save_embeddings.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset citys \
    --data [your dataset path]/cityscapes/ \
    --save-dir ./ \
    --pretrained [your checkpoint path]/deeplabv3_resnet18_citys_best_model.pth

python tsne_visual.py