import torch
from JL_DCF import build_model
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


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


# 卷积可视化
def show_featuremap(model, image):
    # 定义网格
    img_grid = vutils.make_grid(image, normalize=True, scale_each=True, nrow=2)

    # 绘制原始图像
    writer.add_image('raw img', img_grid)  # j 表示feature map数

    model.eval()
    for name, layer in model._modules.items():
        print(name, layer)
        if not ('c' in name):
            return
        image = layer(image)
        if 'c' in name:
            x1 = image.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
            writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)


model = build_model()
model.load_state_dict(torch.load('1.pth'))
# model.eval()
writer = SummaryWriter('/Result')
img_rgb = cv2.imread('./1.jpg')  # 读取图像
img_depth = cv2.imread('./2.png')  # 读取图像
img_rgb, img_depth = np.float32(cv2.resize(img_rgb, (224, 224))) / 255, \
                     np.float32(cv2.resize(img_depth, (224, 224))) / 255  # 为了丢到vgg16要求的224*224 先进行缩放并且归一化
img_rgb = preprocess_image(img_rgb)
img_depth = preprocess_image(img_depth)
img = torch.cat((img_rgb, img_depth), dim=0)
show_featuremap(model, img)

