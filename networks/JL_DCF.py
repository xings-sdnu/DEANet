import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as bn
from resnet import ResNet, Bottleneck
import numpy as np

k = 64


class JLModule(nn.Module):
    def __init__(self, backbone):
        super(JLModule, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(1024, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        # 使用膨胀卷积
        cp.append(nn.Sequential(nn.Conv2d(2048, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

    # 加载预训练模型
    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        self.vgg_conv1.load_state_dict(torch.load('pretrained/vgg_conv1.pth'), strict=True)

    def forward(self, x):
        # put tensor from Resnet backbone to compress model
        feature_extract = []
        feature_extract.append(self.CP[0](self.vgg_conv1(x)))
        x = self.backbone(x)
        for i in range(5):
            feature_extract.append(self.CP[i + 1](x[i]))
        return feature_extract  # list of tensor that compress model output


# CM（跨模态融合方法）
class CMLayer(nn.Module):
    def __init__(self):
        super(CMLayer, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3,
                              bias=False)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.channel = ChannelAttention(64)
        self.spatial = SpatialAttention()
        self.aspp = _ASPP(64, 64, [1, 3, 5, 7])

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            # RGB信息
            part1 = list_x[i][0]
            # 深度信息
            part2 = list_x[i][1]
            part1 = self.aspp(part1.unsqueeze(dim=0))

            part2 = self.bn(part2.unsqueeze(dim=0))
            part2 = self.relu(part2)
            part2 = self.maxpool(part2)

            part2_1 = self.channel(part2)
            part2_1 = part2.mul(part2_1)
            part2_1 = self.spatial(part2_1)
            part2_1 = part2_1.mul(part2_1)
            part2 = part2 + part2_1
            # sum = part1 + part2
            sum = (part1 + part2 + (part1 * part2))
            resl.append(sum)

            # JL-DCF:CM模块
            # 对RGB和Depth进行融合，并在第一个维度扩展一个维度
            # sum = (part1 + part2 + (part1 * part2)).unsqueeze(dim=0)
            # print("sum.shape", sum.shape)  # [1, 64, 20, 20])
            # resl.append(sum)

            # # 定义卷积，将连接的傅里叶变换后的深度图像数据维度变为和卷积后的图片一样
            # conv = nn.Conv2d(sal_depth.shape[1] * 2, list_x[i].shape[1], kernel_size=3, stride=1, padding=1)
            # conv = conv.cuda()
            # # 傅里叶变换
            # fu_depth = fft.fftn(sal_depth)
            # # 取实部
            # depth_real = torch.real(fu_depth)
            # # 取虚部
            # depth_imag = torch.imag(fu_depth)
            # # 虚部实部相结合
            # fu_depth = torch.cat([depth_real, depth_imag], dim=1)  # torch.Size([1, 6, 320, 320])
            #
            # # 反傅里叶变换
            # fu_depth = fft.ifftn(fu_depth)
            # # 取出实部
            # fu_depth = torch.real(fu_depth)
            # # print("sal_depth.shape:", sal_depth.shape)
            # fu_depth = conv(fu_depth)
            # # resize傅里叶变换后的深度图像大小，和卷积后的深度图像保持一致
            # fu_depth = F.interpolate(fu_depth, (part2.shape[3], part2.shape[3]),
            #                          mode='bilinear', align_corners=True)
            # print("fu_depth:", fu_depth.shape)
            # print("part2:", part2.shape)
            # 深度图经过卷积和傅里叶变换后元素相乘，获得互补信息
            # part2 = part2.mul(fu_depth)
        return resl


class _ASPP(nn.Module):
    """
        Atrous spatial pyramid pooling (ASPP) deeplabv2
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


# 特征聚合模块
class FAModule(nn.Module):
    def __init__(self):
        super(FAModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1*1
        self.conv_branch1 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu)
        # 1*1 + 3*3
        self.conv_branch2 = nn.Sequential(nn.Conv2d(k, int(k / 2), 1), self.relu,
                                          nn.Conv2d(int(k / 2), int(k / 4), 3, 1, 1), self.relu)
        # 1*1 + 3*3 + 3*3
        self.conv_branch3 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu,
                                          nn.Conv2d(int(k / 4), int(k / 4), 3, 1, 1), self.relu,
                                          nn.Conv2d(int(k / 4), int(k / 4), 3, 1, 1), self.relu)
        # maxpool + 1*1
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(k, int(k / 4), 1), self.relu)

        # self.conv_branch1 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu)
        # self.conv_branch2 = nn.Sequential(nn.Conv2d(k, int(k / 2), 1), self.relu,
        #                                   nn.Conv2d(int(k / 2), int(k / 4), 3, 1, 1), self.relu)
        # self.conv_branch3 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu,
        #                                   nn.Conv2d(int(k / 4), int(k / 4), 5, 1, 2), self.relu)
        # self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(k, int(k / 4), 1), self.relu)

    def forward(self, x_cm, x_fa):
        # element-wise addition
        x = x_cm

        for i in range(len(x_fa)):
            x += x_fa[i]
        # aggregation
        x_branch1 = self.conv_branch1(x)
        x_branch2 = self.conv_branch2(x)
        x_branch3 = self.conv_branch3(x)
        x_branch4 = self.conv_branch4(x)

        x_cat = torch.cat((x_branch1, x_branch2, x_branch3, x_branch4), dim=1)
        return x_cat


class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            # 上采样操作
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


class JL_DCF(nn.Module):
    def __init__(self, base_model_cfg, JLModule, cm_layers, feature_aggregation_module, JL_score_layers,
                 DCF_score_layers, upsampling):
        super(JL_DCF, self).__init__()
        self.base_model_cfg = base_model_cfg
        # 加载骨架网络：ResNet
        self.JLModule = JLModule
        # 特征聚合
        self.FA = nn.ModuleList(feature_aggregation_module)
        # 上采样
        self.upsampling = nn.ModuleList(nn.ModuleList(upsampling[i]) for i in range(0, 4))
        # JL部分最后得到的预测图像
        self.score_JL = JL_score_layers
        # DCF最后得到的预测图像
        self.score_DCF = DCF_score_layers
        # 跨模态融合模块
        self.cm = cm_layers

    def forward(self, x):
        # 首先加载网络，进行卷积操作
        x = self.JLModule(x)
        # 跨模态融合
        x_cm = self.cm(x)
        # 得到第六次卷积后的预测图，得到全局指导loss
        s_coarse = self.score_JL(x[5])
        # 将顺序颠倒:[320,320,64]
        x_cm = x_cm[::-1]
        # print(len(x_cm))
        x_fa = []
        x_fa_temp = []
        # 将CM5和CM6聚合成为FA5
        x_fa.append(self.FA[4](x_cm[1], x_cm[0]))
        x_fa_temp.append(x_fa[0])
        for i in range(len(x_cm) - 2):
            for j in range(len(x_fa)):
                x_fa_temp[j] = self.upsampling[i][i - j](x_fa[j])
            x_fa.append(self.FA[3 - i](x_cm[i + 2], x_fa_temp))
            x_fa_temp.append(x_fa[-1])
        # 取出最后一个作为最终预测图，同时得到final loss
        s_final = self.score_DCF(x_fa[-1])
        return s_final, s_coarse


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 维度注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def build_model(base_model_cfg='resnet'):
    feature_aggregation_module = []
    for i in range(5):
        feature_aggregation_module.append(FAModule())
    upsampling = []
    for i in range(0, 4):
        upsampling.append([])
        for j in range(0, i + 1):
            upsampling[i].append(
                nn.ConvTranspose2d(k, k, kernel_size=2 ** (j + 2), stride=2 ** (j + 1), padding=2 ** (j)))
    if base_model_cfg == 'resnet':
        backbone = ResNet(Bottleneck, [3, 4, 23, 3])
        return JL_DCF(base_model_cfg, JLModule(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                      ScoreLayer(k), upsampling)
