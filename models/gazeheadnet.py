import torch.nn as nn
import torch
from torchvision import models
import numpy as np


class GazeHeadNet(nn.Module):
    def __init__(self):
        super(GazeHeadNet, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.features
        self.FC1 = nn.Linear(512, 64, bias=True)
        self.FC2 = nn.Linear(64, 64, bias=True)
        self.FC3 = nn.Linear(64, 4, bias=True)
        self.act = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        nn.init.kaiming_normal_(self.FC1.weight.data)
        nn.init.constant_(self.FC1.bias.data, val=0)
        nn.init.kaiming_normal_(self.FC2.weight.data)
        nn.init.constant_(self.FC2.bias.data, val=0)
        nn.init.kaiming_normal_(self.FC3.weight.data)
        nn.init.constant_(self.FC3.bias.data, val=0)

    def forward(self, X, use_feature=False):
        feature = []
        h = self.vgg16[:1](X)
        feature.append(h.view(X.shape[0], -1))
        h = self.vgg16[1:3](h)
        feature.append(h.view(X.shape[0], -1))
        h = self.vgg16[3:6](h)
        feature.append(h.view(X.shape[0], -1))
        h = self.vgg16[6:8](h)
        feature.append(h.view(X.shape[0], -1))
        h = self.vgg16[8:11](h)
        feature.append(h.view(X.shape[0], -1))
        h = self.vgg16[11:](h)
        h = h.mean(-1).mean(-1)
        h = self.act(self.FC1(h))
        h = self.act(self.FC2(h))
        h = self.tanh(self.FC3(h))
        h = np.pi * 0.5 * h
        gaze_hat = h[:, :2]
        head_hat = h[:, 2:]
        if use_feature:
            return feature, gaze_hat, head_hat
        return gaze_hat, head_hat

