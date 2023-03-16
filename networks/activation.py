# coding: utf-8
import os
from PIL import Image
from JL_DCF import build_model
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = build_model()
model.load_state_dict(torch.load('1.pth'))


def preprocess_image(img):
    """
    预处理层
    将图像进行标准化处理
    """
    mean = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]  # BGR > RGB

    # 标准化处理， 将bgr三层都处理
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))  # transpose HWC > CHW
    preprocessed_img = torch.from_numpy(preprocessed_img)  # totensor
    preprocessed_img.unsqueeze_(0)
    input = torch.tensor(preprocessed_img, requires_grad=True)

    return input


img_rgb = cv2.imread('./1.jpg')  # 读取图像
img_depth = cv2.imread('./2.png')  # 读取图像
img_rgb, img_depth = np.float32(cv2.resize(img_rgb, (224, 224))) / 255, \
                     np.float32(cv2.resize(img_depth, (224, 224))) / 255  # 为了丢到vgg16要求的224*224 先进行缩放并且归一化
img_rgb = preprocess_image(img_rgb)
img_depth = preprocess_image(img_depth)
img = torch.cat((img_rgb, img_depth), dim=0)
# 定义一个辅助函数，获取指定层名称的特征
activation = {}  # 保存不同层的输出


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# 获取中间层的卷积后的图像特征
model.eval()
model.cm.register_forward_hook(get_activation("spatial"))
_ = model(img)
maxpool1 = activation["spatial"]

# 对中间层进行可视化，可视化64个特征映射
plt.figure(figsize=(11, 6))
for i in range(maxpool1.shape[1]):
    # 可视化每张手写体
    plt.subplot(6, 11, i + 1)
    plt.imshow(maxpool1.data.numpy()[0, i, :, :], cmap="gray")
    plt.axis("off")
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("经过注意力层的特征映射")
plt.show()