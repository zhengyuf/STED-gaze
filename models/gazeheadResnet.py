import torch.nn as nn
import torch
from torchvision import models
import numpy as np


class GazeHeadResNet(nn.Module):
    def __init__(self):
        super(GazeHeadResNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=4, bias=True)

    def forward(self, X):
        h = self.resnet50(X)
        gaze_hat = h[:, :2]
        head_hat = h[:, 2:]
        return gaze_hat, head_hat

