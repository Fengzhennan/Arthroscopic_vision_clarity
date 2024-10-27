import torch
import torch.nn as nn
from torchvision import models

class TV_Loss(nn.Module):
    def __init__(self, weight: float = 1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
        tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)

class featuremap(nn.Module):
    def __init__(self, requires_grad=False):
        super(featuremap, self).__init__()
        vgg = models.vgg19(pretrained=True).cuda() # .cuda()
        # vgg.load_state_dict(torch.load(r'E:\lianxi\vgg19-dcbb9e9d.pth'))
        vgg.eval()
        vgg_pretrained_features = vgg.features

        self.requires_grad = requires_grad

        self.slice_conv1 = torch.nn.Sequential()
        self.slice_conv2 = torch.nn.Sequential()
        self.slice_conv3 = torch.nn.Sequential()
        self.slice_conv4 = torch.nn.Sequential()
        self.slice_conv5 = torch.nn.Sequential()

        self.slice_conv1.add_module("conv1_2", vgg_pretrained_features[:3])
        self.slice_conv2.add_module("conv2_2", vgg_pretrained_features[:8])
        self.slice_conv3.add_module("conv3_4", vgg_pretrained_features[:17])
        self.slice_conv4.add_module("conv4_4", vgg_pretrained_features[:26])
        self.slice_conv5.add_module("conv5_4", vgg_pretrained_features[:35])

        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        feature_map1 = self.slice_conv1(x)
        feature_map2 = self.slice_conv2(x)
        feature_map3 = self.slice_conv3(x)
        feature_map4 = self.slice_conv4(x)
        feature_map5 = self.slice_conv5(x)
        feature_map = [feature_map1, feature_map2, feature_map3, feature_map4, feature_map5]
        return feature_map

class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.featuremap = featuremap().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [0.1, 0.1, 1.0, 1.0, 1.0]  # 权重

    def forward(self, x, gt):  # 显示x_vgg[0]的64个特征图
        x_featuremap = self.featuremap(x)
        gt_featuremap = self.featuremap(gt)

        perceptual_loss_list = [self.criterion(x, gt) * w for x, gt, w in zip(x_featuremap, gt_featuremap, self.weights)]

        perceptual_loss = sum(perceptual_loss_list)

        return perceptual_loss

class Style_loss(nn.Module):
    def __init__(self):
        super(Style_loss, self).__init__()
        self.featuremap = featuremap().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [0.1, 0.1, 1.0, 1.0, 1.0]  # 权重

    def forward(self, x, gt):  # 显示x_vgg[0]的64个特征图
        x_featuremap = self.featuremap(x)
        gt_featuremap = self.featuremap(gt)

        style_loss_list = [self.criterion(self._gram_mat(x), self._gram_mat(gt)) for x, gt in zip(x_featuremap, gt_featuremap)]

        style_loss = sum(style_loss_list)

        return style_loss

    def _gram_mat(self, x):
        """
        Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram