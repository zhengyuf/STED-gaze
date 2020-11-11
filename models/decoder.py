# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello, Yufeng Zheng.
# --------------------------------------------------------
import numpy as np

import torch
import torch.nn as nn

from .densenet import DenseNetBlock, DenseNetTransitionUp, DenseNetDecoderLastLayers
from core import DefaultConfig

config = DefaultConfig()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):

    def __init__(self, num_all_embedding_features):
        super(Decoder, self).__init__()

        # Define feature map dimensions at bottleneck
        self.bottleneck_shape = (2, 8) if config.densenet_blocks == 4 else (2, 2)
        decoder_input_c = int(num_all_embedding_features / np.prod(self.bottleneck_shape))
        self.decoder_input_c = decoder_input_c
        self.decoder = DenseNetDecoder(
            self.decoder_input_c,
            num_blocks=config.densenet_blocks,
            compression_factor= 1.0,
        )
        self.use_fc = num_all_embedding_features != decoder_input_c * np.prod(self.bottleneck_shape)
        if self.use_fc:
            self.fc_dec = nn.Linear(num_all_embedding_features, decoder_input_c * np.prod(self.bottleneck_shape))

    def forward(self, embeddings):
        x = torch.cat([e.reshape(e.shape[0], -1) for e in embeddings], dim=-1)
        if self.use_fc:
            x = self.fc_dec(x)
        x = x.view(-1, self.decoder_input_c, *self.bottleneck_shape)
        x = self.decoder(x)
        return x


class DenseNetDecoder(nn.Module):

    def __init__(self, c_in, num_blocks=4, num_layers_per_block=4,
                 p_dropout=0.0, compression_factor=1.0,
                 activation_fn=nn.LeakyReLU, normalization_fn=nn.InstanceNorm2d):
        super(DenseNetDecoder, self).__init__()


        c_to_concat = [0] * (num_blocks + 2)

        assert (num_layers_per_block % 2) == 0
        c_now = c_in
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=num_layers_per_block,
                growth_rate=config.growth_rate,
                p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
                transposed=True,
            ))
            c_now = list(self.children())[-1].c_now

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionUp(
                    c_now, p_dropout=p_dropout,
                    compression_factor=compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now
                c_now += c_to_concat[i]

        # Last up-sampling conv layers
        self.last = DenseNetDecoderLastLayers(c_now,
                                              growth_rate=config.growth_rate,
                                              activation_fn=activation_fn,
                                              normalization_fn=normalization_fn,
                                              )
        self.c_now = 1

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            x = module(x)
        return x

