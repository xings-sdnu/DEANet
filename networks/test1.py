# coding: utf-8
import os
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
from JL_DCF import build_model


model_ft = models.resnet101(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, class_num)
model_ft.load_state_dict(torch.load('./ant_bee_model.pth', map_location=lambda  storage, loc:storage))

model_features = nn.Sequential(*list(model_ft.children())[:-2])
fc_weights = model_ft.state_dict()['fc.weight'].cpu().numpy()  #[2,2048]  numpy数组取维度fc_weights[0].shape->(2048,)
class_ = {0:'ant', 1:'bee'}
model_ft.eval()
model_features.eval()

fc = torch.load("fc.pkl", map_location=lambda storage, loc: storage)
fc.eval()
#fc = nn.Linear(num_ftrs, class_num)   #如果没有保存fc参数，则可以用这一行替代
out = torch.nn.functional.adaptive_avg_pool2d(output, (1,1))	#补充
out  = 	torch.flatten(out, 1)              #补充
logit = fc(out)  #补充
#img_path = './bee.jpg'             #单张测试
_, img_name = os.path.split(img_path)
features_blobs = []
img = Image.open(img_path).convert('RGB')
img_tensor = transforms['val'](img).unsqueeze(0) #[1,3,224,224]
features = model_features(img_tensor).detach().cpu().numpy()  #[1,2048,7,7]
logit = model_ft(img_tensor)  #[1,2] -> [ 3.3207, -2.9495]
h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  #tensor([0.9981, 0.0019])

probs, idx = h_x.sort(0, True)      #按概率从大到小排列
probs = probs.cpu().numpy()  #if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
idx = idx.cpu().numpy()  #[1, 0]
for i in range(2):
    print('{:.3f} -> {}'.format(probs[i], class_[idx[i]]))  #打印预测结果
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,7,7
    output_cam = []
    for idx in class_idx:  #只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h*w))
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  #(2048, ) * (2048, 7*7) -> (7*7, ) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)

        #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam
CAMs = returnCAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM
print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])
img = cv2.imread(img_path)
height, width, _ = img.shape  #读取输入图片的尺寸
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM resize match input image size
result = heatmap * 0.3 + img * 0.5    #比例可以自己调节

text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100) 				 #激活图结果上的文字显示
cv2.putText(result, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
            color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
CAM_RESULT_PATH = r'D:/Pycode/hymenoptera/CAM/'   #CAM结果的存储地址
if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)
image_name_ = img_name.split(".")[-2]
cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg', result)  #写入存储磁盘

