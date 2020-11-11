"""Copyright 2020 ETH Zurich, Yufeng Zheng, Seonwook Park
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from .encoder import Encoder
from .decoder import Decoder
from .discriminator import PatchGAN
import losses
from .PerceptualSimilarity.lpips_models import PerceptualLoss
from .gazeheadnet import GazeHeadNet
from .gazeheadResnet import GazeHeadResNet
from core import DefaultConfig
import numpy as np
import torch.nn.functional as F
config = DefaultConfig()


class STED(nn.Module):

    def __init__(self):
        super(STED, self).__init__()
        self.configuration = []
        if config.size_0d_unit > 0:
            self.configuration += [(0, config.size_0d_unit)]
        self.configuration += ([(1, config.size_1d_unit)] * config.num_1d_units +
                               [(2, config.size_2d_unit)] * config.num_2d_units)

        self.num_all_pseudo_labels = np.sum([dof for dof, _ in self.configuration])
        self.num_all_embedding_features = np.sum([
            (dof + 1) * num_feats for dof, num_feats in self.configuration])
        self.encoder = Encoder(self.num_all_pseudo_labels, self.num_all_embedding_features, self.configuration).to("cuda")
        self.decoder = Decoder(self.num_all_embedding_features).to("cuda")
        self.lpips = PerceptualLoss(model='net-lin', net='alex').to("cuda")
        self.GazeHeadNet_train = GazeHeadNet().to("cuda")
        self.GazeHeadNet_eval = GazeHeadResNet().to("cuda")
        self.discriminator = PatchGAN(input_nc=3).to('cuda')
        self.generator_params = []
        for name, param in self.named_parameters():
            if 'encoder' in name or 'decoder' in name or 'classifier' in name:
                self.generator_params.append(param)

        for param in self.GazeHeadNet_train.parameters():
            param.requires_grad = False
        for param in self.GazeHeadNet_eval.parameters():
            param.requires_grad = False
        for param in self.lpips.parameters():
            param.requires_grad = False

        self.generator_optimizer = optim.Adam(self.generator_params, lr=config.lr, weight_decay=config.l2_reg)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.lr, weight_decay=config.l2_reg)
        self.optimizers = [self.generator_optimizer, self.discriminator_optimizer]
        # Wrap optimizer instances with AMP
        if config.use_apex:
            from apex import amp
            models = [self.encoder, self.decoder, self.discriminator, self.GazeHeadNet_eval, self.GazeHeadNet_train, self.lpips]
            models, self.optimizers = amp.initialize(models, self.optimizers, opt_level='O1', num_losses=len(self.optimizers))
            [self.generator_optimizer, self.discriminator_optimizer] = self.optimizers
            [self.encoder, self.decoder, self.discriminator, self.GazeHeadNet_eval, self.GazeHeadNet_train, self.lpips] = models

    def rotation_matrix_1d(self, pseudo_label, inverse=False):
        cos = torch.cos(pseudo_label)
        sin = torch.sin(pseudo_label)
        matrices = torch.stack([cos, -sin, sin, cos], dim=1).view(-1, 2, 2)
        if inverse:
            matrices = torch.transpose(matrices, 1, 2)
        return matrices

    rot2d_ones = None
    rot2d_zeros = None

    def rotation_matrix_2d(self, pseudo_label, inverse=False):
        cos = torch.cos(pseudo_label)
        sin = torch.sin(pseudo_label)
        if (self.rot2d_ones is None
                or self.rot2d_zeros is None
                or self.rot2d_ones.shape[0] != pseudo_label.shape[0]):
            self.rot2d_ones = torch.ones_like(cos[:, 0])
            self.rot2d_zeros = torch.zeros_like(cos[:, 0])
        ones = self.rot2d_ones
        zeros = self.rot2d_zeros
        matrices_1 = torch.stack([ones, zeros, zeros,
                                  zeros, cos[:, 0], -sin[:, 0],
                                  zeros, sin[:, 0], cos[:, 0]
                                  ], dim=1)
        matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                                  zeros, ones, zeros,
                                  -sin[:, 1], zeros, cos[:, 1]
                                  ], dim=1)
        matrices_1 = matrices_1.view(-1, 3, 3)
        matrices_2 = matrices_2.view(-1, 3, 3)
        matrices = torch.matmul(matrices_2, matrices_1)
        if inverse:
            matrices = torch.transpose(matrices, 1, 2)
        return matrices

    def rotate(self, embeddings, pseudo_labels, rotate_or_not=None, inverse=False):
        rotation_matrices = []
        for (dof, _), pseudo_label in zip(self.configuration, pseudo_labels):
            if dof == 0:
                assert pseudo_label is None
                rotation_matrix = None
            elif dof == 1:
                rotation_matrix = self.rotation_matrix_1d(pseudo_label, inverse=inverse)
            elif dof == 2:
                rotation_matrix = self.rotation_matrix_2d(pseudo_label, inverse=inverse)
            else:
                raise ValueError('Invalid DoF: %d' % dof)
            rotation_matrices.append(rotation_matrix)
        rotated_embeddings = [
            torch.matmul(rotation_matrix, embedding)
            if rotation_matrix is not None else embedding
            for embedding, rotation_matrix in zip(embeddings, rotation_matrices)
        ]

        if rotate_or_not is not None:
            mixed_embeddings = [rotated_embeddings[i] * rotate_or_not[i] + embeddings[i] * (1 - rotate_or_not[i])
                                for i in range(len(embeddings))]
            return mixed_embeddings
        return rotated_embeddings

    def forward(self, data):
        output_dict = OrderedDict()
        losses_dict = OrderedDict()
        # Encode input from a
        pseudo_labels_a, embeddings_a = self.encoder(data['image_a'])
        losses_dict['gaze_a'] = losses.gaze_angular_loss(y=data['gaze_a'], y_hat=pseudo_labels_a[-1])
        losses_dict['head_a'] = losses.gaze_angular_loss(y=data['head_a'], y_hat=pseudo_labels_a[-2])
        # Direct reconstruction
        image_a_rec = self.decoder(embeddings_a)
        gaze_a_rec, head_a_rec = self.GazeHeadNet_eval(image_a_rec)

        # embedding disentanglement error
        idx = 0
        gaze_disentangle_loss = 0
        head_disentangle_loss = 0
        batch_size = data['image_a'].shape[0]
        num_0d_units = 1 if config.size_0d_unit > 0 else 0
        for dof, num_feats in self.configuration:
            if dof != 0:
                random_angle = (torch.rand(batch_size, dof).to("cuda") - 0.5) * np.pi * 0.2
                random_angle += pseudo_labels_a[idx]
                if dof == 2:
                    rotated_embedding = torch.matmul(self.rotation_matrix_2d(random_angle, False), torch.matmul(
                        self.rotation_matrix_2d(pseudo_labels_a[idx], True), embeddings_a[idx]))
                else:
                    rotated_embedding = torch.matmul(self.rotation_matrix_1d(random_angle, False), torch.matmul(
                        self.rotation_matrix_1d(pseudo_labels_a[idx], True), embeddings_a[idx]))
                new_embedding = [item for item in embeddings_a]
                new_embedding[idx] = rotated_embedding
                image_random = self.decoder(new_embedding)
                gaze_random, head_random = self.GazeHeadNet_eval(image_random)
                if idx < config.num_1d_units + config.num_2d_units + num_0d_units - 2:
                    gaze_disentangle_loss += losses.gaze_angular_loss(gaze_a_rec, gaze_random)
                    head_disentangle_loss += losses.gaze_angular_loss(head_a_rec, head_random)
                if idx == config.num_1d_units + config.num_2d_units + num_0d_units - 2:  # head
                    losses_dict['head_to_gaze'] = losses.gaze_angular_loss(gaze_a_rec, gaze_random)
                if idx == config.num_1d_units + config.num_2d_units + num_0d_units - 1:  # gaze
                    losses_dict['gaze_to_head'] = losses.gaze_angular_loss(head_a_rec, head_random)
            idx += 1
        if config.num_1d_units + config.num_2d_units - 2 != 0:
            losses_dict['gaze_disentangle'] = gaze_disentangle_loss / (
                        config.num_1d_units + config.num_2d_units - 2)
            losses_dict['head_disentangle'] = head_disentangle_loss / (
                        config.num_1d_units + config.num_2d_units - 2)

        # Calculate some errors if target image is available
        if 'image_b' in data:
            # redirect with pseudo-labels
            gaze_embedding = torch.matmul(self.rotation_matrix_2d(data['gaze_b'], False),
                                          torch.matmul(self.rotation_matrix_2d(pseudo_labels_a[-1], True),
                                                       embeddings_a[-1]))
            head_embedding = torch.matmul(self.rotation_matrix_2d(data['head_b'], False),
                                          torch.matmul(self.rotation_matrix_2d(pseudo_labels_a[-2], True),
                                                       embeddings_a[-2]))
            embeddings_a_to_b = embeddings_a[:-2]
            embeddings_a_to_b.append(head_embedding)
            embeddings_a_to_b.append(gaze_embedding)
            output_dict['image_b_hat'] = self.decoder(embeddings_a_to_b)
            gaze_b_hat, head_b_hat = self.GazeHeadNet_eval(output_dict['image_b_hat'])
            losses_dict['head_redirection'] = losses.gaze_angular_loss(y=data['head_b'], y_hat=head_b_hat)
            losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=data['gaze_b'], y_hat=gaze_b_hat)

            losses_dict['lpips'] = torch.mean(self.lpips(data['image_b'], output_dict['image_b_hat']))
            losses_dict['l1'] = losses.reconstruction_l1_loss(data['image_b'], output_dict['image_b_hat'])

            pseudo_labels_b, _ = self.encoder(data['image_b'])
            normalized_embeddings_from_a = self.rotate(embeddings_a, pseudo_labels_a, inverse=True)
            embeddings_a_to_b_all = self.rotate(normalized_embeddings_from_a, pseudo_labels_b)
            output_dict['image_b_hat_all'] = self.decoder(embeddings_a_to_b_all)
            losses_dict['lpips_all'] = torch.mean(self.lpips(data['image_b'], output_dict['image_b_hat_all']))
            losses_dict['l1_all'] = losses.reconstruction_l1_loss(data['image_b'], output_dict['image_b_hat_all'])

        return output_dict, losses_dict

    def optimize(self, data, current_step):
        if config.use_apex:
            from apex import amp
        losses_dict = OrderedDict()
        for param in self.discriminator.parameters():
            param.requires_grad = True
        for param in self.generator_params:
            param.requires_grad = True

        pseudo_labels_a, embeddings_a = self.encoder(data['image_a'])
        pseudo_labels_b, embeddings_b = self.encoder(data['image_b'])

        if config.use_mixing:
            num_0d_units = 1 if config.size_0d_unit > 0 else 0
            random = np.random.randint(2, size=[num_0d_units + config.num_1d_units + config.num_2d_units,
                                                config.batch_size, 1, 1]).tolist()
            random_tensor = torch.tensor(random, dtype=torch.float, requires_grad=False).to("cuda")

            normalized_embeddings_from_a_mix = self.rotate(embeddings_a, pseudo_labels_a, random_tensor, inverse=True)
            embeddings_a_to_mix = self.rotate(normalized_embeddings_from_a_mix, pseudo_labels_b, random_tensor)
            image_mix_hat = self.decoder(embeddings_a_to_mix)

            pseudo_labels_mix_hat, embeddings_mix_hat = self.encoder(image_mix_hat)

        # a -> b
        normalized_embeddings_from_a = self.rotate(embeddings_a, pseudo_labels_a, inverse=True)
        embeddings_a_to_b = self.rotate(normalized_embeddings_from_a, pseudo_labels_b)
        image_b_hat = self.decoder(embeddings_a_to_b)


        # optimize discriminator
        real = self.discriminator(data['image_b'])
        fake = self.discriminator(image_b_hat.detach())

        losses_dict['discriminator'] = losses.discriminator_loss(real=real, fake=fake)
        losses_dict['generator'] = losses.generator_loss(fake=fake)
        discriminator_loss = losses_dict['discriminator'] * config.coeff_discriminator_loss
        # Warm up period for generator losses
        losses_dict['discrim_coeff'] = torch.tensor(max(min(1.0, current_step / 20000.0), 0.0))

        self.discriminator_optimizer.zero_grad()
        if config.use_apex:
            with amp.scale_loss(discriminator_loss, self.discriminator_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            discriminator_loss.backward()
        self.discriminator_optimizer.step()

        for param in self.discriminator.parameters():
            param.requires_grad = False

        # for generator update

        losses_dict['l1'] = losses.reconstruction_l1_loss(x=data['image_b'], x_hat=image_b_hat)
        total_loss = losses_dict['l1'] * config.coeff_l1_loss
        if not config.semi_supervised:
            losses_dict['gaze_a'] = (losses.gaze_angular_loss(y=data['gaze_a'], y_hat=pseudo_labels_a[-1]) +
                                 losses.gaze_angular_loss(y=data['gaze_b'], y_hat=pseudo_labels_b[-1]))/2
            losses_dict['head_a'] = (losses.gaze_angular_loss(y=data['head_a'], y_hat=pseudo_labels_a[-2]) +
                                 losses.gaze_angular_loss(y=data['head_b'], y_hat=pseudo_labels_b[-2]))/2
        else:
            losses_dict['gaze_a'] = losses.gaze_angular_loss(y=data['gaze_a'], y_hat=pseudo_labels_a[-1])
            losses_dict['head_a'] = losses.gaze_angular_loss(y=data['head_a'], y_hat=pseudo_labels_a[-2])

            losses_dict['gaze_a_unlabeled'] = losses.gaze_angular_loss(y=data['gaze_b'], y_hat=pseudo_labels_b[-1])
            losses_dict['head_a_unlabeled'] = losses.gaze_angular_loss(y=data['head_b'], y_hat=pseudo_labels_b[-2])

        total_loss += (losses_dict['gaze_a'] + losses_dict['head_a']) * config.coeff_gaze_loss


        fake = self.discriminator(image_b_hat)
        generator_loss = losses.generator_loss(fake=fake)
        total_loss += generator_loss * config.coeff_discriminator_loss * losses_dict['discrim_coeff']

        if config.coeff_embedding_consistency_loss != 0:
            normalized_embeddings_from_a = self.rotate(embeddings_a, pseudo_labels_a, inverse=True)
            normalized_embeddings_from_b = self.rotate(embeddings_b, pseudo_labels_b, inverse=True)
            flattened_normalized_embeddings_from_a = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_a], dim=1)
            flattened_normalized_embeddings_from_b = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_b], dim=1)
            losses_dict['embedding_consistency'] = (1.0 - torch.mean(
                F.cosine_similarity(flattened_normalized_embeddings_from_a,
                                    flattened_normalized_embeddings_from_b, dim=-1)))
            total_loss += losses_dict['embedding_consistency'] * config.coeff_embedding_consistency_loss

        if config.coeff_disentangle_embedding_loss != 0:
            assert config.use_mixing is True
            flattened_before_c = torch.cat([
                e.reshape(e.shape[0], -1) for e in embeddings_a_to_mix], dim=1)
            flattened_after_c = torch.cat([
                e.reshape(e.shape[0], -1) for e in embeddings_mix_hat], dim=1)
            losses_dict['embedding_disentangle'] = (1.0 - torch.mean(
                F.cosine_similarity(flattened_before_c,
                                    flattened_after_c, dim=-1)))
            total_loss += losses_dict['embedding_disentangle'] * config.coeff_disentangle_embedding_loss
        if config.coeff_disentangle_pseudo_label_loss != 0:
            assert config.use_mixing is True
            losses_dict['label_disentangle'] = 0
            pseudo_labels_a_b_mix = []
            for i in range(len(pseudo_labels_a)):  # pseudo code
                if pseudo_labels_b[i] is not None:
                    pseudo_labels_a_b_mix.append(
                        pseudo_labels_b[i] * random_tensor[i].squeeze(-1) + pseudo_labels_a[i] * (1 - random_tensor[i].squeeze(-1)))
                else:
                    pseudo_labels_a_b_mix.append(None)

            for y, y_hat in zip(pseudo_labels_a_b_mix[-2:], pseudo_labels_mix_hat[-2:]):
                if y is not None:
                    losses_dict['label_disentangle'] += losses.gaze_angular_loss(y, y_hat)
            total_loss += losses_dict['label_disentangle'] * config.coeff_disentangle_pseudo_label_loss

        feature_h, gaze_h, head_h = self.GazeHeadNet_train(image_b_hat, True)
        feature_t, gaze_t, head_t = self.GazeHeadNet_train(data['image_b'], True)
        losses_dict['redirection_feature_loss'] = 0
        for i in range(len(feature_h)):
            losses_dict['redirection_feature_loss'] += nn.functional.mse_loss(feature_h[i], feature_t[i].detach())
        total_loss += losses_dict['redirection_feature_loss'] * config.coeff_redirection_feature_loss
        losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=gaze_t.detach(), y_hat=gaze_h)
        total_loss += losses_dict['gaze_redirection'] * config.coeff_redirection_gaze_loss
        losses_dict['head_redirection'] = losses.gaze_angular_loss(y=head_t.detach(), y_hat=head_h)
        total_loss += losses_dict['head_redirection'] * config.coeff_redirection_gaze_loss
        self.generator_optimizer.zero_grad()
        if config.use_apex:
            with amp.scale_loss(total_loss, [self.generator_optimizer]) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        self.generator_optimizer.step()

        return losses_dict, image_b_hat

    def latent_walk(self, data):
        walk_len = 40
        batch_size = data['image_a'].shape[0]
        pseudo_labels_a, embeddings_a = self.encoder(data['image_a'])
        pseudo_labels_b, embeddings_b = self.encoder(data['image_b'])
        num_0d_unit = 1 if config.size_0d_unit > 0 else 0
        walks = []
        walk = []
        embeddings_copy = embeddings_a.copy()
        # interpolation of appearance code
        for j in np.linspace(-1, 1, num=walk_len, endpoint=True):
            embed = embeddings_a[0] * (1 - j) + embeddings_b[0] * j
            embeddings_copy[0] = embed
            image_ = self.decoder(embeddings_copy)
            walk.append(image_)
        walks.append(walk)
        # latent walk of 1d factors
        for i in range(config.num_1d_units):
            walk = []
            embeddings_copy = embeddings_a.copy()
            for j in np.linspace(-np.pi / 2 * 0.9, np.pi / 2 * 0.9, num=walk_len, endpoint=True):
                angle = torch.tensor(j, dtype=torch.float).unsqueeze(0).expand(batch_size, -1).to("cuda")
                embed = torch.matmul(self.rotation_matrix_1d(angle, False), torch.matmul(
                    self.rotation_matrix_1d(pseudo_labels_a[num_0d_unit + i], True), embeddings_a[num_0d_unit + i]))
                embeddings_copy[num_0d_unit + i] = embed
                image_ = self.decoder(embeddings_copy)
                walk.append(image_)
            walks.append(walk)
        # interpolation of 2d factors
        walk_2d = np.stack(
            [
                # Circular motion
                # thetas
                0.6 * np.sin(np.linspace(0, 2 * np.pi, num=walk_len, endpoint=False)),
                # phis
                0.6 * np.cos(np.linspace(0, 2 * np.pi, num=walk_len, endpoint=False)),
                # # Criss-cross motion
                # # thetas
                # list(np.linspace(0, min_v, num=int(walk_len / 8), endpoint=False)) +
                # list(np.linspace(min_v, max_v, num=int(walk_len / 4), endpoint=False)) +
                # list(np.linspace(max_v, 0, num=int(walk_len / 8), endpoint=False)) +
                # list(np.zeros(int(walk_len / 2))),
                #
                # # phis
                # list(np.zeros(int(walk_len / 2))) +
                # list(np.linspace(0, min_v, num=int(walk_len / 8), endpoint=False)) +
                # list(np.linspace(min_v, max_v, num=int(walk_len / 4), endpoint=False)) +
                # list(np.linspace(max_v, 0, num=int(walk_len / 8), endpoint=False)),
            ],
            axis=-1,
        )
        num_0d_1d_unit = num_0d_unit + config.num_1d_units
        for i in range(config.num_2d_units):
            walk = []
            embeddings_copy = embeddings_a.copy()
            for j in walk_2d:
                angle = torch.tensor(j, dtype=torch.float).unsqueeze(0).expand(batch_size, -1).to("cuda")
                embed = torch.matmul(self.rotation_matrix_2d(angle, False), torch.matmul(
                    self.rotation_matrix_2d(pseudo_labels_a[num_0d_1d_unit + i], True),
                    embeddings_a[num_0d_1d_unit + i]))
                embeddings_copy[num_0d_1d_unit + i] = embed
                image_ = self.decoder(embeddings_copy)
                walk.append(image_)
            walks.append(walk)
        return walks

    def redirect(self, data):
        output_dict = OrderedDict()
        pseudo_labels_a, embeddings_a = self.encoder(data['image_a'])

        gaze_embedding = torch.matmul(self.rotation_matrix_2d(data['gaze_b_r'], False),
                                  torch.matmul(self.rotation_matrix_2d(pseudo_labels_a[-1], True),
                                               embeddings_a[-1]))
        head_embedding = torch.matmul(self.rotation_matrix_2d(data['head_b_r'], False),
                                  torch.matmul(self.rotation_matrix_2d(pseudo_labels_a[-2], True),
                                               embeddings_a[-2]))
        embeddings_a_to_b = embeddings_a[:-2]
        embeddings_a_to_b.append(head_embedding)
        embeddings_a_to_b.append(gaze_embedding)
        output_dict['image_b_hat_r'] = self.decoder(embeddings_a_to_b)
        return output_dict

    def clean_up(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
