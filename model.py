import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import time
import os
import numpy as np
# import cv2
import random
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import init
import time

# from NMS.nms.gpu_nms import gpu_nms
# from pth_nms_test import nms
from torchvision.ops import RoIAlign
from torchvision.ops.boxes import nms as torch_nms


# from torchvision.ops import nms

def get_merge_bbox(dets, inds):
    xx1 = np.min(dets[inds][:, 0])
    yy1 = np.min(dets[inds][:, 1])
    xx2 = np.max(dets[inds][:, 2])
    yy2 = np.max(dets[inds][:, 3])
    return np.array((xx1, yy1, xx2, yy2))


def pth_nms_merge(dets, thresh, topk):
    dets = dets.cpu().data.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    boxes_merge = []
    cnt = 0
    while order.size > 0:
        i = order[0]

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]

        inds_merge = np.where((ovr > 0.5) * (0.9 * scores[i] < scores[order[1:]]))[0]
        boxes_merge.append(get_merge_bbox(dets, np.append(i, order[inds_merge + 1])))
        order = order[inds + 1]

        cnt += 1
        if cnt >= topk:
            break

    return torch.from_numpy(np.array(boxes_merge))


#############################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


###################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimpleFPA(nn.Module):
    def __init__(self, in_planes, out_planes):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(SimpleFPA, self).__init__()

        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

        # Global pooling branch
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        out = x_master + x_gpb

        return out



class PyramidFeatures(nn.Module):
    """Feature pyramid module with top-down feature pathway"""

    def __init__(self, B2_size, B3_size, B4_size, B5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = SimpleFPA(B5_size, feature_size)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(B4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(B3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        B3, B4, B5 = inputs

        P5_x = self.P5_1(B5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(B4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(B3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class PyramidAttentions(nn.Module):  ###缺陷很小，特征目标不大，无法突出相关的目标。

    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(PyramidAttentions, self).__init__()

        self.A3_1 = SpatialGate(channel_size)
        self.A3_2 = ChannelGate(channel_size)

        self.A4_1 = SpatialGate(channel_size)
        self.A4_2 = ChannelGate(channel_size)

        self.A5_1 = SpatialGate(channel_size)
        self.A5_2 = ChannelGate(channel_size)

    def forward(self, inputs):
        F3, F4, F5 = inputs

        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A3 = A3_spatial * F3 + A3_channel * F3

        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_spatial * F4 + A4_channel * F4

        A5_spatial = self.A5_1(F5)
        # test = A5_spatial[0]
        # test = torch.unsqueeze(test, dim=0)
        # test = F.interpolate(test, size=(768, 768), mode="bilinear")
        # test = torch.squeeze(test, dim=0)
        # test = torch.squeeze(test, dim=0)
        # test = test.cpu().detach().numpy()
        # a = np.ones_like(test)*255
        # test = np.multiply(a,test)
        # test = test.astype(np.uint8)
        # cv2.imshow("test", test)
        A5_channel = self.A5_2(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_spatial * F5 + A5_channel * F5

        return [A3, A4, A5, A3_spatial, A4_spatial, A5_spatial]


class SpatialGate(nn.Module):
    """generation spatial attention mask"""

    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)


class ChannelGate(nn.Module):
    """generation channel attention mask"""

    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def generate_anchors_single_pyramid(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (x1, y1, x2, y2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return torch.from_numpy(boxes)


class ResNet(nn.Module):
    """implementation of AP-CNN on ResNet"""

    def __init__(self, num_classes, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.num_classes == 200:
            hidden_num = 512
        else:
            hidden_num = 256

        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels,
                         self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        self.apn = PyramidAttentions(channel_size=256)
        # freeze(self)

        self.cls5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(256, hidden_num),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(256, hidden_num),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  ##还是要多所有的特征做averaging pooling.
            Flatten(),
            nn.Linear(256, hidden_num),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls_concate = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 3, hidden_num),
            nn.Linear(hidden_num, self.num_classes)
        )


        self.cls5_roi = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(256, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls4_roi = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(256, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls3_roi = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  ##还是要多所有的特征做averaging pooling.
            Flatten(),
            nn.Linear(256, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls_concate_roi = nn.Sequential(
            Flatten(),
            nn.Linear(256*3, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls5_sum = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(256*2, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls4_sum = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(256*2, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls3_sum = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  ##还是要多所有的特征做averaging pooling.
            Flatten(),
            nn.Linear(256*2, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.cls_concate_sum = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 3 * 2, hidden_num),
            nn.ELU(inplace=True),
            nn.Linear(hidden_num, self.num_classes)
        )

        self.Conv_roi_1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn_roi_1 = nn.BatchNorm2d(32)
        self.Conv_roi_2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.roi_align = RoIAlign(output_size=(192, 192), spatial_scale=1, sampling_ratio=-1)
        # self.criterion = nn.CrossEntropyLoss()
        self.roiLoss = True  # 是否对roi的分类结果进行loss计算
        self.mul_cls = True  # 单分类或者多分类

        self.criterion = nn.BCEWithLogitsLoss()

        self.inplanes = 64
        self.conv1_roi = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_roi = nn.BatchNorm2d(64)
        self.relu_roi = nn.ReLU(inplace=True)
        self.maxpool_roi = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_roi = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_roi = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_roi = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_roi = self._make_layer(block, 512, layers[3], stride=2)

        self.fpn_roi = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        self.apn_roi = PyramidAttentions(channel_size=256)
        self.line1 = nn.Linear(self.num_classes, self.num_classes, bias=False)
        self.line2 = nn.Linear(self.num_classes, self.num_classes, bias=False)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def Concate(self, f3, f4, f5):
        f3 = nn.AdaptiveAvgPool2d(output_size=1)(f3)
        f5 = nn.AdaptiveAvgPool2d(output_size=1)(f5)
        f4 = nn.AdaptiveAvgPool2d(output_size=1)(f4)
        f_concate = torch.cat([f3, f4, f5], dim=1)
        return f_concate


    def forward(self, inputs, target, TRAIN, roi_train):
        # ResNet backbone with FC removed
        n, c, img_h, img_w = inputs.size()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # stage I
        f3, f4, f5 = self.fpn([x2, x3, x4])
        f3_att, f4_att, f5_att, a3, a4, a5 = self.apn([f3, f4, f5])

        # feature concat
        f_concate = self.Concate(f3, f4, f5)
        out_concate = self.cls_concate(f_concate)


        out3 = self.cls3(f3_att)
        out4 = self.cls4(f4_att)
        out5 = self.cls5(f5_att)
        out = (out3 + out4 + out5 + out_concate) / 4



        # out = nn.Sigmoid()(out)


        # 二阶段
        crop_att = torch.ones(f5_att.size()).cuda()
        CAM_info, cam_list = self.get_CAM_ROI_val(f5_att, self.cls5, out, inputs, target)



        crop_att = crop_att - CAM_info
        inputs_crop = inputs


        x = self.conv1_roi(inputs_crop)
        x = self.bn1_roi(x)
        x = self.relu_roi(x)
        x = self.maxpool_roi(x)

        x1_roi = self.layer1_roi(x)
        x2_roi = self.layer2_roi(x1_roi)
        x3_roi = self.layer3_roi(x2_roi)
        x4_roi = self.layer4_roi(x3_roi)

        f3_roi, f4_roi, f5_roi = self.fpn_roi([x2_roi, x3_roi, x4_roi])
        f3_att_roi, f4_att_roi, f5_att_roi, a3, a4, a5 = self.apn_roi([f3_roi, f4_roi, f5_roi])
        f_concate_roi = self.Concate(f3_att_roi, f4_att_roi, f5_att_roi)


        f5_att_roi = f5_att_roi*crop_att
        f4_att_roi = f4_att_roi*F.interpolate(crop_att, size=(f4_att_roi.size()[-2], f4_att_roi.size()[-1]),
                                            mode="bilinear")
        f3_att_roi = f3_att_roi * F.interpolate(crop_att, size=(f3_att_roi.size()[-2], f3_att_roi.size()[-1]),
                                            mode="bilinear")

        out3_roi = self.cls3_roi(f3_att_roi)
        out4_roi = self.cls4_roi(f4_att_roi)
        out5_roi = self.cls5_roi(f5_att_roi)
        out_concate_roi = self.cls_concate_roi(f_concate_roi)

        f3_att_sum = torch.cat([f3_att, f3_att_roi], dim=1)
        f4_att_sum = torch.cat([f4_att, f4_att_roi], dim=1)

        f5_att_sum = torch.cat([f5_att, f5_att_roi], dim=1)

        f_concate_sum = self.Concate(f3_att_sum, f4_att_sum, f5_att_sum)

        out3_sum = self.cls3_sum(f3_att_sum)
        out4_sum = self.cls4_sum(f4_att_sum)
        out5_sum = self.cls5_sum(f5_att_sum)
        out_concate_sum = self.cls_concate_sum(f_concate_sum)

        out_sum = (out3_sum + out4_sum + out5_sum + out_concate_sum) / 4


        loss3_sum = self.criterion(out3_sum, target)
        loss4_sum = self.criterion(out4_sum, target)
        loss5_sum = self.criterion(out5_sum, target)
        loss_concate_sum = self.criterion(out_concate_sum, target)

        loss_sum = loss3_sum + loss4_sum + loss5_sum + loss_concate_sum



        loss3_roi = self.criterion(out3_roi, target)
        loss4_roi = self.criterion(out4_roi, target)
        loss5_roi = self.criterion(out5_roi, target)
        loss_concate_roi = self.criterion(out_concate_roi, target)


        out_roi = (out3_roi + out4_roi + out5_roi + out_concate_roi) / 4

        loss_roi = loss3_roi + loss4_roi + loss5_roi + loss_concate_roi



        loss_last = loss_roi + loss_sum



        out_roi = nn.Sigmoid()(out_roi)
        out = nn.Sigmoid()(out)
        out_last = nn.Sigmoid()(out_sum)

        zero = torch.zeros_like(out_roi)
        one = torch.ones_like(out_roi)
        pred_roi = torch.where(out_roi > 0.5, one, out_roi)
        pred_roi = torch.where(out_roi <= 0.5, zero, pred_roi)

        pred = torch.where(out > 0.5, one, out)
        pred = torch.where(out <= 0.5, zero, pred)

        pred_last = torch.where(out_last > 0.5, one, out_last)
        pred_last = torch.where(out_last <= 0.5, zero, pred_last)

        correct = 0
        correct_roi = 0
        correct_last = 0
        for i in range(n):
            if (pred_roi[i] == target[i]).all(): correct_roi += 1
            if (pred[i] == target[i]).all(): correct += 1
            if (pred_last[i] == target[i]).all(): correct_last += 1



        loss_ret = {'loss': loss_last, 'loss1': loss_roi , 'loss2': loss_sum}
        acc_ret = {'acc': correct_last, 'acc_resnet': correct, 'acc_cam': correct_roi}

        if TRAIN:
            return acc_ret, loss_ret
        else:
            return out_roi, out, out_roi


    def get_CAM_ROI_val(self, x4, net_cls, outs, x2, targets):

            return x_roi_2, cam_list




    def get_img_label(self, x, roi, target):
            return x_roi.cuda(), x_target.cuda(), x_index.cuda()



    def choose_bath(self, roi_out, batch_index):

        return bath_roi








def resnet18(num_classes, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(num_classes, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_woroi_v2(num_classes, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(num_classes, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(num_classes, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# model = resnet50_woroi_v2(num_classes=5)
# for num, m in enumerate(model.app):
#     if num < 14:
#         m.eval()
#         print(m)

# model.cuda()
# intput = torch.randn(2, 3, 768, 768).cuda()
# out = torch.tensor([[0.,0.,0.,0.,1.],[1.,0.,0.,0.,0.]]).cuda()
# b = model(intput, out, True)
# # # print(b)
# # torchinfo.summary(model, input_size=(10,3,448,448))
# print(intput.shape)
# print(out.shape)
# loss_ret, acc_ret, mask_cat, roi_list = model(intput, out)
# print(loss_ret, acc_ret)
