# DEANet-pytorch

Pytorch implementation for Saliency Detection Framework Based on Deep Enhanced Attention Network (ICONIP 2021)]

# Requirements
* Python 3.7 <br>
* Pytorch 1.8.1 <br>
* Torchvision 0.9.1 <br>
* Cuda 11.0

# Usage
This is the Pytorch implementation of DEANet. It has been trained and tested on Windows (Win10 + Cuda 11 + Python 3.7 + Pytorch 1.8.1),
and it should also work on Linux but we didn't try. 

## To Train 
* Download the pre-trained ImageNet backbone (resnet101 and vgg_conv1, whereas the latter already exists in the folder), and put it in the 'pretrained' folder
* Download the training dataset and modify the 'train_root' and 'train_list' in the `main.py`
* Set 'mode' to 'train'
* Run `main.py`

## To Test 
* Download the testing dataset and have it in the 'dataset/test/' folder 
* Download the already-trained JL-DCF pytorch model and modify the 'model' to its saving path in the `main.py`
* Modify the 'test_folder' in the `main.py` to the testing results saving folder you want
* Modify the 'sal_mode' to select one testing dataset (NJU2K, NLPR, STERE, RGBD135, LFSD or SIP)
* Set 'mode' to 'test'
* Run `main.py`

## Learning curve
The training log is saved in the 'log' folder. If you want to see the learning curve, you can get it by using: ` tensorboard --logdir your-log-path`

# Pre-trained ImageNet model for training
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password: rllb<br>

# Trained model for testing
Baidu Pan: [DEANet-pytorch](https://pan.baidu.com/s/1KoxUvnnM5zJoFPEkrv7b1Q), password: jdpb<br>
Google Drive: https://drive.google.com/open?id=12u37yz-031unDPJoKaZ0goK8BtPP-6Cj<br>

# DEANet-pytorch saliency maps
Baidu Pan: [Saliency maps](https://pan.baidu.com/s/1IzAjbbhoAdhsg-2B_gSwqw), password: 4nqr<br>
Google Drive: https://drive.google.com/open?id=1mHMN36aI5zNt50DQBivSDyYvCQ9eeGhP<br>

# Dataset
Baidu Pan:<br>
[Training dataset (with horizontal flip)](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
[Testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
Google Drive:<br>
[Training dataset (with horizontal flip)](https://drive.google.com/open?id=12ais7wZhTjaFO4BHJyYyNuzzM312EWCT)<br>
[Testing datadet](https://drive.google.com/open?id=18ALe_HBuNjVTB_US808d8ZKfpd_mwLy5)<br>

# Performance
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

# Citation
Please cite our paper if you find the work useful:<br>

        @InProceedings{Xing_2021_ICONIP,
        author = {Xing Sheng, Zhuoran Zheng, Qiong Wu, Chunmeng Kang, Yunliang Zhuang, Lei Lyu, Chen Lyu},
        title = {Saliency Detection Framework Based on Deep Enhanced Attention Network},
        booktitle = {International Conference on Neural Information Processing (ICONIP)},
        pages={274--286},
        year = {2021}
        }

# Benchmark RGB-D SOD
The complete RGB-D SOD benchmark can be found in this page  
http://dpfan.net/d3netbenchmark/
