# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello.
# --------------------------------------------------------
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DenseNetInitialLayers(nn.Module):

    def __init__(self, growth_rate=8, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetInitialLayers, self).__init__()
        c_next = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, c_next, bias=False,
                               kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        self.norm = normalization_fn(c_next, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)

        c_out = 4 * growth_rate
        self.conv2 = nn.Conv2d(2 * growth_rate, c_out, bias=False,
                               kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        self.c_now = c_out
        self.c_list = [c_next, c_out]

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        prev_scale_x = x
        x = self.conv2(x)
        return x, prev_scale_x


class DenseNetBlock(nn.Module):

    def __init__(self, c_in, z_dim_app=None, num_layers=4, growth_rate=8, p_dropout=0.1,
                 use_bottleneck=False, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d, transposed=False, use_style=False):
        super(DenseNetBlock, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.use_style = use_style
        c_now = c_in
        for i in range(num_layers):
            i_ = i + 1
            if use_bottleneck:
                self.add_module('bneck%d' % i_, DenseNetCompositeLayer(
                    c_now, 4 * growth_rate, z_dim_app=z_dim_app, kernel_size=1, p_dropout=p_dropout,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                    use_style=use_style,
                ))
            self.add_module('compo%d' % i_, DenseNetCompositeLayer(
                4 * growth_rate if use_bottleneck else c_now, growth_rate, z_dim_app=z_dim_app,
                kernel_size=3, p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                transposed=transposed,
                use_style=use_style,
            ))
            c_now += list(self.children())[-1].c_now
        self.c_now = c_now

    def forward(self, x, apps=None):
        x_before = x
        for i, (name, module) in enumerate(self.named_children()):
            if ((self.use_bottleneck and name.startswith('bneck'))
                    or name.startswith('compo')):
                x_before = x
            if self.use_style:
                x = module(x, apps[:, i])
            else:
                x = module(x)
            if name.startswith('compo'):
                x = torch.cat([x_before, x], dim=1)
        return x


class DenseNetTransitionDown(nn.Module):

    def __init__(self, c_in, compression_factor=0.1, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionDown, self).__init__()
        c_out = int(compression_factor * c_in)
        self.composite = DenseNetCompositeLayer(
            c_in, c_out,
            kernel_size=1, p_dropout=p_dropout,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c_now = c_out

    def forward(self, x):
        x = self.composite(x)
        x = self.pool(x)
        return x


class DenseNetCompositeLayer(nn.Module):

    def __init__(self, c_in, c_out, z_dim_app=None, kernel_size=3, growth_rate=8, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d,
                 transposed=False, use_style=False):
        super(DenseNetCompositeLayer, self).__init__()
        self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.use_style = use_style
        if use_style:
            self.style_mod = ApplyStyle(z_dim_app=z_dim_app, channels=c_in)
        self.act = activation_fn(inplace=True)
        if transposed:
            assert kernel_size > 1
            self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size,
                                           padding=1 if kernel_size > 1 else 0,
                                           stride=1, bias=False).to(device)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1,
                                  padding=1 if kernel_size > 1 else 0, bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x, app=None):
        x = self.norm(x)
        x = self.act(x)
        if self.use_style:
            x = self.style_mod(x, app)
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

class DenseNetDecoderLastLayers(nn.Module):

    def __init__(self, c_in, growth_rate=8, activation_fn=nn.ReLU,
                 normalization_fn=nn.BatchNorm2d):
        super(DenseNetDecoderLastLayers, self).__init__()
        # First deconv
        self.conv1 = nn.ConvTranspose2d(c_in, 4 * growth_rate, bias=False,
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv1.weight.data)

        # Second deconv
        c_in = 4 * growth_rate
        self.norm2 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv2 = nn.ConvTranspose2d(c_in, 2 * growth_rate, bias=False,
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        nn.init.kaiming_normal_(self.conv2.weight.data)

        # Final conv
        c_in = 2*growth_rate
        c_out = 3
        self.norm3 = normalization_fn(c_in, track_running_stats=False).to(device)
        self.conv3 = nn.Conv2d(c_in, c_out, bias=False,
                               kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        self.c_now = c_out

    def forward(self, x, from_enc=None):
        x = self.conv1(x)
        if from_enc is not None:
            x = torch.cat([x, from_enc], 1)
        #
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        #
        x = self.norm3(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class DenseNetTransitionUp(nn.Module):

    def __init__(self, c_in, compression_factor=0.1, p_dropout=0.1,
                 activation_fn=nn.ReLU, normalization_fn=nn.BatchNorm2d):
        super(DenseNetTransitionUp, self).__init__()
        c_out = int(compression_factor * c_in)
        self.norm = normalization_fn(c_in, track_running_stats=False).to(device)
        self.act = activation_fn(inplace=True)
        self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=3,
                                       stride=2, padding=1, output_padding=1,
                                       bias=False).to(device)
        nn.init.kaiming_normal_(self.conv.weight.data)
        self.drop = nn.Dropout2d(p=p_dropout) if p_dropout > 1e-5 else None
        self.c_now = c_out

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
