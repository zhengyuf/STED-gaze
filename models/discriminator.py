import torch.nn as nn
import functools


class PatchGAN(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(PatchGAN, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm1 = nn.BatchNorm2d(ndf * 2)
        self.act = nn.LeakyReLU(0.2, True)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        self.norm2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        self.norm3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=kw, stride=1, padding=padw)  # output 1 channel prediction map

    def forward(self, input):
        """Standard forward."""

        input = self.conv1(input)
        input = self.act(input)

        input = self.conv2(input)
        input = self.norm1(input)
        input = self.act(input)

        input = self.conv3(input)
        input = self.norm2(input)
        input = self.act(input)

        input = self.conv4(input)
        input = self.norm3(input)
        input = self.act(input)

        input = self.conv5(input)

        return input


