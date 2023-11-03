#  Cross-Image Relational Knowledge Distillation for Semantic Segmentation

This repository contains the source code of CIRKD ([Cross-Image Relational Knowledge Distillation for Semantic Segmentation](https://arxiv.org/pdf/2204.06986.pdf)) and implementations of semantic segmentation tasks on some datasets.



## Requirement


Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.8.0

NCCL for CUDA 11.1

Install python packages:
```
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
```

Backbones pretrained on ImageNet:

| CNN | Transformer |
| -- | -- |
|[resnet101-imagenet.pth](https://drive.google.com/file/d/1V8-E4wm2VMsfnNiczSIDoSM7JJBMARkP/view?usp=sharing)| [mit_b0.pth](https://pan.baidu.com/s/1Figp042rc9VNtPc_fkNW3g?pwd=swor )|
|[resnet18-imagenet.pth](https://drive.google.com/file/d/1_i0n3ZePtQuh66uQIftiSwN7QAUlFb8_/view?usp=sharing) | [mit_b1.pth](https://pan.baidu.com/s/1OUblLHQbq18DvXGzRU58jA?pwd=03yb)|
|[mobilenetv2-imagenet.pth](https://drive.google.com/file/d/12EDZjDSCuIpxPv-dkk1vrxA7ka0b0Yjv/view?usp=sharing) | [mit_b4.pth](https://pan.baidu.com/s/1j8pXjZZ-YSi2JXpsaQSSTQ?pwd=cvpd )|


Support datasets:

| Dataset | Train Size | Val Size | Test Size | Class |
| -- | -- | -- |-- |-- |
| Cityscapes | 2975 | 500 | 1525 |19|
| Pascal VOC Aug | 10582 | 1449 | -- | 21 |
| CamVid | 367 | 101 | 233 | 11 |
| ADE20K | 20210 | 2000 | -- | 150 |
| COCO-Stuff-164K | 118287 | 5000 |-- | 182 |

## Performance on Cityscapes

All models are trained over 8 * NVIDIA GeForce RTX 3090

| Role | Network |Method | Val mIoU|test mIoU|Pretrained |train script |
| -- | -- | -- |-- |-- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|78.07 |77.46 |[Google Drive](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/deeplabv3_res101.sh)|
| Student| DeepLabV3-ResNet18|Baseline| 74.21 | 73.45|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/deeplabv3_res18.sh)|
| Student| DeepLabV3-ResNet18|CIRKD| 76.38 |75.05|[Google Drive](https://drive.google.com/file/d/1ebP28XJWJNDbU9OmnfT7x2JJWHWaDFMi/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/deeplabv3_res18.sh)|
| Student| DeepLabV3-MobileNetV2|Baseline|73.12|72.36|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/deeplabv3_mbv2.sh)|
| Student| DeepLabV3-MobileNetV2|CIRKD|75.42|74.03|[Google Drive](https://drive.google.com/file/d/1iw8GXxj612C_nRtBdS72kgIZ5nYOU1Ys/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/deeplabv3_mbv2.sh)|
| Student| PSPNet-ResNet18|Baseline|72.55|72.29|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/pspnet_res18.sh)|
| Student| PSPNet-ResNet18|CIRKD|74.73|74.05|[Google Drive](https://drive.google.com/file/d/1zfpWVfzOpeVG7_WjeQPGB0rDl_XQX8ZG/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/pspnet_res18.sh)|


## Performance of Segmentation KD methods on Cityscapes

| Method | Val mIoU |Val mIoU |Val mIoU | train script |
| -- | -- | -- |-- |-- |
|  Teacher | DeepLabV3-ResNet101| DeepLabV3-ResNet101 | SegFormer-MiT-B4|
|  Baseline | 78.07 | 78.07 | 81.23 [[pretrained]](https://pan.baidu.com/s/1fslpQwIeIJ67veX0Q1WbHg?pwd=x6lf)  | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/segformer_mit_b4.sh)|
|  Student | DeepLabV3-ResNet18| DeepLabV3-MobileNetV2 | SegFormer-MiT-B0||
|  Baseline | 74.21 | 73.12 | 75.58 [[pretrained]](https://pan.baidu.com/s/1pxM6O8upmf0TPSJJaDw6RA?pwd=umzz)  | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/segformer_mit_b0.sh)|
|  SKD [3]| 75.42 | 73.82 | 76.43 [[pretrained]](https://pan.baidu.com/s/1uJ-Q2XUZXELgkOPhVRqLdw?pwd=3xuh) | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_skd.sh)|
|  IFVD [4]| 75.59 | 73.50 |76.30 [[pretrained]](https://pan.baidu.com/s/12X0XGF2ZoS7OCSpXE2Si6A?pwd=m36j) | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_ifvd.sh)|
|  CWD [5]| 75.55 | 74.66 |74.80 [[pretrained]](https://pan.baidu.com/s/1ViKwRw1XbwB9eH96jiSXMA?pwd=aola) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_cwd.sh) |
|  DSD [6]| 74.81 | 74.11 |76.62 [[pretrained]](https://pan.baidu.com/s/1gBoF2OFkN90LODuu7n2F-A?pwd=w5j9) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_dsd.sh) |
|  CIRKD [7]| 76.38 | 75.42 | 76.92 [[pretrained]](https://pan.baidu.com/s/1YeZZ68E3s4bmW3QUODK-wQ?pwd=olh6) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/segformer_mit_b0.sh) |

The references are shown in [references.md](https://github.com/winycg/CIRKD/tree/main/losses/references.md)

### Evaluate pre-trained models on Cityscapes test sets

You can run [test_cityscapes.sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/test_cityscapes.sh).
You can zip the resulting images and submit it to the [Cityscapes test server](https://www.cityscapes-dataset.com/submit/).


**Note**: The current codes have been reorganized and we have not tested them thoroughly. If you have any questions, please contact us without hesitation. 


## Performance of Segmentation KD methods on Pascal VOC

The Pascal VOC dataset for segmentation is available at [Baidu Drive](https://pan.baidu.com/s/1MX2ea7rNRqbDqOKQ8E6XpQ?pwd=d2fp )


| Role | Network |Method | Val mIoU|train script |Pretrained |
| -- | -- | -- |-- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|77.67 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/voc/deeplabv3_res101.sh)|[Google Drive](https://drive.google.com/file/d/1rYTaVq_ooiAI4oFOcDP8K3SpSbjURGnX/view?usp=sharing) |
| Student| DeepLabV3-ResNet18|Baseline| 73.21 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/voc/deeplabv3_res18.sh)||
| Student| DeepLabV3-ResNet18|CIRKD| 74.50 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/voc/deeplabv3_res18.sh)||
| Student| PSPNet-ResNet18|Baseline|73.33|[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/voc/pspnet_res18.sh)||
| Student| PSPNet-ResNet18|CIRKD|74.78 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/voc/pspnet_res18.sh)||

## Performance of Segmentation KD methods on CamVid

The CamVid dataset for segmentation is available at [Baidu Drive](https://pan.baidu.com/s/1Z0h4y1-4k0LP8OCGY_Xixw?pwd=bl12)

| Role | Network |Method | Val mIoU|train script |Pretrained |
| -- | -- | -- |-- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|69.84 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/camvid/deeplabv3_res101.sh)|[Google Drive](https://drive.google.com/file/d/1BK8Flukoz-Mtd0e1iwFG5rLxi_ES76d2/view?usp=sharing) |
| Student| DeepLabV3-ResNet18|Baseline| 66.92 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/camvid/deeplabv3_res18.sh)||
| Student| DeepLabV3-ResNet18|CIRKD| 68.21 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/camvid/deeplabv3_res18.sh)||
| Student| PSPNet-ResNet18|Baseline|66.73|[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/camvid/pspnet_res18.sh)||
| Student| PSPNet-ResNet18|CIRKD|68.65 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/camvid/pspnet_res18.sh)||

## Performance of Segmentation KD methods on ADE20K

The ADE20K dataset for segmentation is available at [Google Drive](https://drive.google.com/file/d/10cCHvCZ3HTxtE9iaSlMcMF0oDgtMpVF8/view?usp=share_link )

| Role | Network |Method | Val mIoU|train script |Pretrained |
| -- | -- | -- |-- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|42.70 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/ade20k/deeplabv3_res101.sh)|[Google Drive](https://drive.google.com/file/d/1jlywjvZqKTUWCEwdFpPKHzrpNgeckMcO/view?usp=share_link ) |
| Student| DeepLabV3-ResNet18|Baseline| 33.91 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/ade20k/deeplabv3_res18.sh)||
| Student| DeepLabV3-ResNet18|CIRKD| 35.41 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/ade20k/deeplabv3_res18.sh)||


## Performance of Segmentation KD methods on COCO-Stuff-164K

| Role | Network |Method | Val mIoU|train script |Pretrained |
| -- | -- | -- |-- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|38.71|[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/coco_stuff_164k/deeplabv3_res101.sh)|[Google Drive](https://drive.google.com/file/d/1uOCl8ZYK22d7D1WXG4tnHc6slT8_iSjk/view?usp=share_link ) |
| Student| DeepLabV3-ResNet18|Baseline| 32.60 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/coco_stuff_164k/deeplabv3_res18.sh)||
| Student| DeepLabV3-ResNet18|CIRKD| 33.11 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/coco_stuff_164k/deeplabv3_res18.sh)||



## Visualization of segmentation mask using pretrained models
| Dataset | Color Pallete | Blend | Scripts |
| -- | -- |-- |-- |
| Pascal VOC |![top1](figures/2007_000033.png) |![top1](figures/2007_000033.jpg) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/visualize/pascal_voc.sh)|
| Cityscapes |![top1](figures/frankfurt_000000_000576_gtFine_labelIds.png) |![top1](figures/frankfurt_000000_000576_leftImg8bit.png) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/visualize/cityscapes.sh)|
| ADE20K |![top1](figures/ADE_val_00000053.png) |![top1](figures/ADE_val_00000053.jpg) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/visualize/ade20k.sh)|
| COCO-Stuff-164K |![top1](figures/000000025394.png) |![top1](figures/000000025394.jpg) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/visualize/coco_stuff_164k.sh)|


## Citation
We would appreciate it if you could give this repo a star or cite our paper!
```
@inproceedings{yang2022cross,
  title={Cross-image relational knowledge distillation for semantic segmentation},
  author={Yang, Chuanguang and Zhou, Helong and An, Zhulin and Jiang, Xue and Xu, Yongjun and Zhang, Qian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12319--12328},
  year={2022}
}
```



