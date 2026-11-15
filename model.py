# -*- coding: utf-8 -*-
"""实验一 多任务模型定义：共享主干 + 物种头 + 状态头"""

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str = 'resnet18', pretrained: bool = True):
    """返回去掉分类层的特征提取主干及其输出维度"""
    if name == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == 'resnet50':
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == 'efficientnet_b0':
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    else:
        raise ValueError('不支持的主干网络: %s' % name)
    return net, feat_dim


class PlantMultiTaskModel(nn.Module):
    """共享主干 + 两个独立分类头"""

    def __init__(self, num_species: int, num_status: int,
                 backbone: str = 'resnet18', pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head_species = nn.Linear(feat_dim, num_species)
        self.head_status = nn.Linear(feat_dim, num_status)

    def forward(self, x):
        feat = self.dropout(self.backbone(x))
        return self.head_species(feat), self.head_status(feat)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


if __name__ == '__main__':
    model = PlantMultiTaskModel(num_species=5, num_status=5)
    x = torch.randn(2, 3, 224, 224)
    out_s, out_st = model(x)
    print('物种输出 shape:', tuple(out_s.shape))
    print('状态输出 shape:', tuple(out_st.shape))
