import torch
from torch import nn
import torch.nn.functional as F
from resnet import ResNet, Bottleneck

k = 64


class DEAModule(nn.Module):
    def __init__(self, backbone):
        super(DEAModule, self).__init__()
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
        cp.append(nn.Sequential(nn.Conv2d(2048, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

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
            part1 = list_x[i][0]
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
            sum = (part1 + part2 + (part1 * part2))
            resl.append(sum)

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


class FIModule(nn.Module):
    def __init__(self):
        super(FIModule, self).__init__()
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

    def forward(self, x_cm, x_fa):
        x = x_cm

        for i in range(len(x_fa)):
            x += x_fa[i]
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
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


class DEANet(nn.Module):
    def __init__(self, base_model_cfg, DEAModule, cm_layers, feature_aggregation_module, RGB_score_layers,
                 Depth_score_layers, upsampling):
        super(DEANet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.DEAModule = DEAModule
        self.FI = nn.ModuleList(feature_aggregation_module)
        self.upsampling = nn.ModuleList(nn.ModuleList(upsampling[i]) for i in range(0, 4))
        self.score_RGB = RGB_score_layers
        self.score_Depth = Depth_score_layers
        self.cm = cm_layers

    def forward(self, x):
        x = self.DEAModule(x)
        x_cm = self.cm(x)
        s_coarse = self.score_RGB(x[5])
        x_cm = x_cm[::-1]
        x_fi = []
        x_fi_temp = []
        x_fi.append(self.FI[4](x_cm[1], x_cm[0]))
        x_fi_temp.append(x_fi[0])
        for i in range(len(x_cm) - 2):
            for j in range(len(x_fi)):
                x_fi_temp[j] = self.upsampling[i][i - j](x_fi[j])
            x_fi.append(self.FI[3 - i](x_cm[i + 2], x_fi_temp))
            x_fi_temp.append(x_fi[-1])
        s_final = self.score_Depth(x_fi[-1])
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
        feature_aggregation_module.append(FIModule())
    upsampling = []
    for i in range(0, 4):
        upsampling.append([])
        for j in range(0, i + 1):
            upsampling[i].append(
                nn.ConvTranspose2d(k, k, kernel_size=2 ** (j + 2), stride=2 ** (j + 1), padding=2 ** (j)))
    if base_model_cfg == 'resnet':
        backbone = ResNet(Bottleneck, [3, 4, 23, 3])
        return DEANet(base_model_cfg, DEAModule(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                      ScoreLayer(k), upsampling)
