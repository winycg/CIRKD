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
|  Teacher | DeepLabV3-ResNet101|-|78.07 |77.46 |[Google Drive](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/deeplabv3_res101.sh)|
| Student| DeepLabV3-ResNet18|Baseline| 74.21 | 73.45|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/deeplabv3_res18.sh)|
| Student| DeepLabV3-ResNet18|CIRKD| 76.38 |75.05|[Google Drive](https://drive.google.com/file/d/1ebP28XJWJNDbU9OmnfT7x2JJWHWaDFMi/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/deeplabv3_res18.sh)|
| Student| DeepLabV3-MobileNetV2|Baseline|73.12|72.36|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/deeplabv3_mbv2.sh)|
| Student| DeepLabV3-MobileNetV2|CIRKD|75.42|74.03|[Google Drive](https://drive.google.com/file/d/1iw8GXxj612C_nRtBdS72kgIZ5nYOU1Ys/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/deeplabv3_mbv2.sh)|
| Student| PSPNet-ResNet18|Baseline|72.55|72.29|- |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/citys/pspnet_res18.sh)|
| Student| PSPNet-ResNet18|CIRKD|74.73|74.05|[Google Drive](https://drive.google.com/file/d/1zfpWVfzOpeVG7_WjeQPGB0rDl_XQX8ZG/view?usp=sharing) |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/citys/pspnet_res18.sh)|


## Performance of Segmentation KD methods on Cityscapes

| Method | Val mIoU |Val mIoU | train script |
| -- | -- | -- |-- |
|  Teacher | DeepLabV3-ResNet101| DeepLabV3-ResNet101 | |
|  Baseline | 78.07 | 78.07 |  |
|  Student | DeepLabV3-ResNet18| DeepLabV3-MobileNetV2 | |
|  Baseline | 74.21 | 73.12 |  |
|  SKD [3]| 75.42 | 73.82 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_skd.sh)|
|  IFVD [4]| 75.59 | 73.50 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_ifvd.sh)|
|  CWD [5]| 75.55 | 74.66 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_cwd.sh) |
|  DSD [6]| 74.81 | 74.11 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_kd/train_dsd.sh) |
|  CIRKD [7]| 76.38 | 75.42 |  |

The references are shown in [references.md](https://github.com/winycg/CIRKD/tree/main/losses/references.md)

### Evaluate pre-trained models on Cityscapes test sets

You can run [test_cityscapes.sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/test_cityscapes.sh).
You can zip the resulting images and submit it to the [Cityscapes test server](https://www.cityscapes-dataset.com/submit/).


**Note**: The current codes have been reorganized and we have not tested them thoroughly. If you have any questions, please contact us without hesitation. 


## Performance of Segmentation KD methods on Pascal VOC

The Pascal VOC dataset for segmentation is available at [Baidu Drive](https://pan.baidu.com/s/1MX2ea7rNRqbDqOKQ8E6XpQ?pwd=d2fp )


| Role | Network |Method | Val mIoU|train script |
| -- | -- | -- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|77.67 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/voc/deeplabv3_res101.sh)|
| Student| DeepLabV3-ResNet18|Baseline| 73.21 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/voc/deeplabv3_res18.sh)|
| Student| DeepLabV3-ResNet18|CIRKD| 74.50 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/voc/deeplabv3_res18.sh)|
| Student| PSPNet-ResNet18|Baseline|73.33|[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/voc/pspnet_res18.sh)|
| Student| PSPNet-ResNet18|CIRKD|74.78 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/voc/pspnet_res18.sh)|

## Performance of Segmentation KD methods on CamVid

The CamVid dataset for segmentation is available at [Baidu Drive](https://pan.baidu.com/s/1Z0h4y1-4k0LP8OCGY_Xixw?pwd=bl12)

| Role | Network |Method | Val mIoU|train script |
| -- | -- | -- |-- |-- |
|  Teacher | DeepLabV3-ResNet101|-|69.84 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/camvid/deeplabv3_res101.sh)|
| Student| DeepLabV3-ResNet18|Baseline| 66.92 | [sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/camvid/deeplabv3_res18.sh)|
| Student| DeepLabV3-ResNet18|CIRKD| 68.21 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/camvid/deeplabv3_res18.sh)|
| Student| PSPNet-ResNet18|Baseline|66.73|[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_baseline/camvid/pspnet_res18.sh)|
| Student| PSPNet-ResNet18|CIRKD|68.65 |[sh](https://github.com/winycg/CIRKD/tree/main/train_scripts/train_cirkd/camvid/pspnet_res18.sh)|


## Citation

```
@inproceedings{yang2022cross,
  title={Cross-image relational knowledge distillation for semantic segmentation},
  author={Yang, Chuanguang and Zhou, Helong and An, Zhulin and Jiang, Xue and Xu, Yongjun and Zhang, Qian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12319--12328},
  year={2022}
}
```



