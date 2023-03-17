# DEANet-pytorch

Pytorch implementation for [Saliency Detection Framework Based on Deep Enhanced Attention Network](https://link.springer.com/chapter/10.1007/978-3-030-92273-3_23)(ICONIP 2021)

<!-- TOC -->

- [DEANet in PyTorch](#EdgeSeg-in-pytorch)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [Training](#training)
  - [Test](#test)
  - [Learning Curve](#learning-curve)
  - [Pre-trained ImageNet model for training](#pre-trained-imagenet-model-for-training)
  - [Trained model for testing](#trained-model-for-testing)
  - [DEANet-pytorch saliency maps](#deanet-pytorch-saliency-maps)
  - [Dataset](#dataset)
  - [Performance](#performance)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)


## Requirements
* Python 3.7 <br>
* Pytorch 1.8.1 <br>
* Torchvision 0.9.1 <br>
* Cuda 11.0

# Usage
This is the Pytorch implementation of DEANet. It has been trained and tested on Windows (Win10 + Cuda 11 + Python 3.7 + Pytorch 1.8.1),
and it should also work on Linux but we didn't try. 

## Training
* Download the pre-trained ImageNet backbone (resnet101, Baidu YunPan: [resnet101](https://pan.baidu.com/s/1VEID4futYro5Up2Fbh4HoQ), password:93q7, and put it in the 'pretrained' folder
* Download the training dataset and modify the 'train_root' and 'train_list' in the `main.py`
* Set 'mode' to 'train'
* Run `main.py`

## Test 
* Download the testing dataset and have it in the 'dataset/test/' folder 
* Download the already-trained DEANet pytorch model and modify the 'model' to its saving path in the `main.py`
* Modify the 'test_folder' in the `main.py` to the testing results saving folder you want
* Modify the 'sal_mode' to select one testing dataset (NJU2K, NLPR, STERE, RGBD135, LFSD or SIP)
* Set 'mode' to 'test'
* Run `main.py`

## Learning curve
The training log is saved in the 'log' folder. If you want to see the learning curve, you can get it by using: ` tensorboard --logdir your-log-path`

## Pre-trained ImageNet model for training
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password: rllb<br>

## Trained model for testing
Baidu Pan: [DEANet-pytorch](https://pan.baidu.com/s/103_1FWvTqWygi8e7XLGgSw), password: svyr<br>
Google Drive: <br>

## DEANet-pytorch saliency maps
Baidu Pan: [Saliency maps](https://pan.baidu.com/s/1XncwFhgNLxISZ5kIM1eY8w), password: maft<br>
Google Drive: <br>

## Dataset
Baidu Pan:<br>
[Training dataset (with horizontal flip)](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
[Testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
Google Drive:<br>
[Training dataset (with horizontal flip)](https://drive.google.com/open?id=12ais7wZhTjaFO4BHJyYyNuzzM312EWCT)<br>
[Testing datadet](https://drive.google.com/open?id=18ALe_HBuNjVTB_US808d8ZKfpd_mwLy5)<br>

## Performance
Below is the performance of DEANet-pyotrch (Pytorch implementation). Due to the randomness in the training process, the obtained results will fluctuate slightly.

| Datasets | Metrics | Pytorch |
| -------- | ------- | ------- |
| NJU2K    |S-measure| 0.917   |
|          | maxF    | 0.900   |
|          | maxE    | 0.919   |
|          | MAE     | 0.038   |
| NLPR     |S-measure| 0.959   |
|          | maxF    | 0.922   |
|          | maxE    | 0.979   |
|          | MAE     | 0.014   |
| STERE    |S-measure| 0.908   |
|          | maxF    | 0.877   |
|          | maxE    | 0.921   |
|          | MAE     | 0.041   |
| RGBD135  |S-measure| 0.932   |
|          | maxF    | 0.907   |
|          | maxE    | 0.968   |
|          | MAE     | 0.021   |
| LFSD     |S-measure| 0.855   |
|          | maxF    | 0.855   |
|          | maxE    | 0.885   |
|          | MAE     | 0.078   |
| SSD      |S-measure| 0.870   |
|          | maxF    | 0.830   |
|          | maxE    | 0.901   |
|          | MAE     | 0.051   |  

## Citation
Please cite our paper if you find the work useful:<br>

        @InProceedings{Xing_2021_ICONIP,
        author = {Xing Sheng, Zhuoran Zheng, Qiong Wu, Chunmeng Kang, Yunliang Zhuang, Lei Lyu, Chen Lyu},
        title = {Saliency Detection Framework Based on Deep Enhanced Attention Network},
        booktitle = {International Conference on Neural Information Processing (ICONIP)},
        pages={274--286},
        year = {2021}
        }



## Acknowledgement
+ [JL-DCF](https://github.com/jiangyao-scu/JL-DCF-pytorch)
