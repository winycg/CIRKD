#  Cross-Image Relational Knowledge Distillation for Semantic Segmentation

This repository contains the source code of CIRKD ([Cross-Image Relational Knowledge Distillation for Semantic Segmentation](https://arxiv.org/pdf/2204.06986.pdf)).



## Requirement


Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.8.0

NCCL for CUDA 11.1

Backbones pretrained on ImageNet:
* [resnet101-imagenet.pth](https://drive.google.com/file/d/1V8-E4wm2VMsfnNiczSIDoSM7JJBMARkP/view?usp=sharing) 
* [resnet18-imagenet.pth](https://drive.google.com/file/d/1_i0n3ZePtQuh66uQIftiSwN7QAUlFb8_/view?usp=sharing) 
* [mobilenetv2-imagenet.pth](https://drive.google.com/file/d/12EDZjDSCuIpxPv-dkk1vrxA7ka0b0Yjv/view?usp=sharing) 

## Performance on Cityscapes

All models are trained over 8 * NVIDIA GeForce RTX 3090

| Role | Network |Method | Val mIoU|test mIoU|Pretrained |train script |
| -- | -- | -- |-- |-- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|78.07 |77.46 |[Google Drive](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/deeplabv3_res101.sh)|
| Student| DeepLabV3-ResNet18|Baseline| 74.21 | 73.45|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/deeplabv3_res18.sh)|
| Student| DeepLabV3-ResNet18|CIRKD| 76.38 |75.05|[Google Drive](https://drive.google.com/file/d/1ebP28XJWJNDbU9OmnfT7x2JJWHWaDFMi/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/deeplabv3_res18.sh)|
| Student| DeepLabV3-MobileNetV2|Baseline|73.12|72.36|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/deeplabv3_mbv2.sh)|
| Student| DeepLabV3-MobileNetV2|CIRKD|75.42|74.03|[Google Drive](https://drive.google.com/file/d/1iw8GXxj612C_nRtBdS72kgIZ5nYOU1Ys/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/deeplabv3_mbv2.sh)|
| Student| PSPNet-ResNet18|Baseline|72.55|72.29|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/deeplabv3_mbv2.sh)|
| Student| PSPNet-ResNet18|CIRKD|74.73|74.05|[Google Drive](https://drive.google.com/file/d/1zfpWVfzOpeVG7_WjeQPGB0rDl_XQX8ZG/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/pspnet_res18.sh)|


## Performance of Segmentation KD methods on Cityscapes

| Method | Val mIoU |Val mIoU | train script |
| -- | -- | -- |-- |
|  Teacher | DeepLabV3-ResNet101| DeepLabV3-ResNet101 | |
|  Baseline | 78.07 | 78.07 |  |
|  Student | DeepLabV3-ResNet18| DeepLabV3-MobileNetV2 | |
|  Baseline | 74.21 | 73.12 |  |
|  AT [2] | TODO | TODO |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_at.sh)|
|  SKD [3]| 75.42 | 73.82 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_skd.sh)|
|  IFVD [4]| 75.59 | 73.50 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_ifvd.sh)|
|  CWD [5]| 75.55 | 74.66 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_cwd.sh) |
|  CIRKD [6]| 76.38 | 75.42 |  |

The references are shown in [references.md](https://github.com/winycg/CIRKD/tree/main/losses/references.md)

## Evaluate pre-trained models on Cityscapes val and test sets

### Evaluate the pre-trained models on val set
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m torch.distributed.launch --nproc_per_node=8 eval.py \
  --model deeplabv3 \
  --backbone resnet101 \
  --data [your dataset path]/cityscapes/ \
  --save-dir [your directory path to store log files] \
  --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth
```

### Generate the resulting images on test set
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -m torch.distributed.launch --nproc_per_node=4 test.py \
  --model deeplabv3 \
  --backbone resnet101 \
  --data [your dataset path]/cityscapes/ \
  --save-dir [your directory path to store resulting images] \
  --save-pred \
  --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth
```
You can submit the resulting images to the [Cityscapes test server](https://www.cityscapes-dataset.com/submit/).


The current codes have been reorganized and we have not tested them thoroughly. If you have any questions, please contact us without hesitation. 

## Citation

```
@inproceedings{yang2022cross,
  title={Cross-Image Relational Knowledge Distillation for Semantic Segmentation},
  author={Chuanguang Yang, Helong Zhou, Zhulin An, Xue Jiang, Yongjun Xu, Qian Zhang},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```



