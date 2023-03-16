import torch
from torch.nn import functional as F
from networks.JL_DCF import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from loss import edge_loss
from torchsummary import summary

writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
size_coarse = (20, 20)


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.build_model()
        self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model))
        if config.mode == 'train':
            if self.config.load == '':
                # 加载预训练模型：vgg_conv1.pth
                net = self.net.JLModule.load_pretrained_model(self.config.pretrained_model)  # load pretrained backbone
            else:
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(name)
        # print(model)
        # print("The number of parameters: {}".format(num_params))
        # summary(model, input_size=(3, 320, 320))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')

    def test(self):
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)
                # 将三通道RGB图与变换后的深度图（深度图通过颜色映射转换为3通道）进行串联
                input = torch.cat((images, depth), dim=0)
                preds, pred_coarse = self.net(input,depth)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        # 迭代次数
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        # 初始化优化器
        self.optimizer.zero_grad()
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    # sal_label:真值图像,将开始读取tensor变量复制一份到device指定的GPU上去
                    sal_image, sal_depth, sal_label = sal_image.to(device), sal_depth.to(device), sal_label.to(device)
                # 上采样，使用双线性插值法，将真值图像扩展到指定大小
                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                # 将两个真值图像进行串联，形成batch
                sal_label_coarse = torch.cat((sal_label_coarse, sal_label_coarse), dim=0)
                # 将RGB图与Depth图串联，形成batch，batch结构：320*320*3*2，便于并行处理
                sal_input = torch.cat((sal_image, sal_depth), dim=0)
                sal_final, sal_coarse = self.net(sal_input)
                # 使用二元交叉熵损失函数，全局引导损失 reduction（字符串，可选）–指定要应用于输出的减少量：
                # 'none'| 'mean'| 'sum'。'none'：不应用缩减，
                # 'mean'：输出的总和除以输出中元素的数量，'sum'：输出的总和。注意：size_average
                # 和reduce正在弃用的过程中，同时，指定这两个args中的任何一个将覆盖reduction。默认：'mean'
                sal_loss_coarse = F.binary_cross_entropy_with_logits(sal_coarse, sal_label_coarse, reduction='sum')
                # 全局最终损失
                sal_loss_final = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                # 边缘保持损失
                edges_loss = edge_loss(sal_final, sal_label)
                # 使用λ作为全局引导损失的权重，三项加权和作为最终损失
                sal_loss_fuse = sal_loss_final + 256 * sal_loss_coarse + 10 * edges_loss
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                sal_loss.backward()

                # accumulate gradients as done in DSS
                aveGrad += 1
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss / (self.show_every / self.iter_size)))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    r_sal_loss = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
