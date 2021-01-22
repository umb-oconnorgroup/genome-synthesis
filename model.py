import math
from typing import Callable

import numpy as np
import torch
from torch import nn


def conv1(in_planes, out_planes, stride=1):
    """size 1 kernel convolution with 0 padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv9(in_planes, out_planes, stride=1):
    """size 9 kernel convolution with 4 padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride, padding=4, bias=False)

def conv17(in_planes, out_planes, stride=1):
    """size 17 kernel convolution with 8 padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=17, stride=stride, padding=8, bias=False)

def deconv8(in_planes, out_planes, stride=1):
    """size 8 kernel convolution with 2 padding"""
    return nn.ConvTranspose1d(in_planes, out_planes, kernel_size=8, stride=stride, padding=2, bias=False)

def deconv16(in_planes, out_planes, stride=1):
    """size 16 kernel convolution with 4 padding"""
    return nn.ConvTranspose1d(in_planes, out_planes, kernel_size=16, stride=stride, padding=4, bias=False)


class EncoderBlock(nn.Module):
    """docstring for EncoderBlock"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(EncoderBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv9(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv9(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    """docstring for DecoderBlock"""
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(DecoderBlock, self).__init__()
        # Both self.deconv1 and self.upsample layers upsample the input when stride != 1
        self.conv1 = conv9(inplanes, inplanes)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = deconv8(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.deconv1(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out
        

class Encoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, layers, latent, num_classes, feature_size):
        super(Encoder, self).__init__()

        self.class_embedding = nn.Embedding(num_classes, feature_size)
        self.inplanes = 32
        self.conv1 = conv17(2, self.inplanes, stride=4)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=6, stride=4, padding=2)
        self.layer1 = self._make_layer(32, layers[0])
        self.layer2 = self._make_layer(64, layers[1], stride=4)
        self.layer3 = self._make_layer(128, layers[2], stride=4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, latent)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(conv1(self.inplanes, planes, stride), nn.BatchNorm1d(planes))
        layers = []
        layers.append(EncoderBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(EncoderBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x.unsqueeze(1), self.class_embedding(c).unsqueeze(1)], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        

class Decoder(nn.Module):
    """docstring for Decoder"""
    def __init__(self, layers, latent, num_classes):
        super(Decoder, self).__init__()

        self.class_embedding = nn.Embedding(num_classes, latent)
        self.inplanes = 64
        self.fc = nn.Linear(latent * 2, 128)
        self.deconv1 = deconv16(128, 64, stride=8)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, layers[0], stride=4)
        self.layer2 = self._make_layer(16, layers[1], stride=4)
        self.layer3 = self._make_layer(8, layers[2], stride=4)
        self.deconv2 = deconv8(8, 1, stride=4)
        self.bn2 = nn.BatchNorm1d(2048)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(deconv8(self.inplanes, planes, stride), nn.BatchNorm1d(planes))
        layers = []
        for _ in range(blocks - 1):
            layers.append(EncoderBlock(self.inplanes, self.inplanes))
        layers.append(DecoderBlock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, c):
        x = torch.cat([x, self.class_embedding(c)], 1)
        x = self.fc(x).unsqueeze(-1)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.deconv2(x).squeeze(1)
        x = self.bn2(x)
        return x


class CVAE(nn.Module):
    """docstring for CVAE"""
    def __init__(self, latent, num_classes, feature_size):
        super(CVAE, self).__init__()
        self.latent = latent
        layers = [2, 2, 1]
        self.encoder = Encoder(layers, latent * 2, num_classes, feature_size)
        layers = [2, 2, 2]
        self.decoder = Decoder(layers, latent, num_classes)

    def encode(self, x, c):
        z = self.encoder(x, c)
        z_mu, z_var = z.split(self.latent, 1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp()
            eps = torch.randn_like(std)
            return eps*std + mu
        else:
            return mu

    def decode(self, z, c):
        return self.decoder(z, c)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class CHVAE(nn.Module):
    """docstring for CHVAE"""
    def __init__(self, feature_size, latent_size, class_size, hidden_size_1=400, hidden_size_2=200):
        super(CHVAE, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.class_size = class_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.fc1 = nn.Linear(feature_size + class_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2 + latent_size)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, latent_size)

        self.fc4 = nn.Linear(latent_size // 2 + class_size, hidden_size_2)
        self.bn4 = nn.BatchNorm1d(hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2 + latent_size // 2, hidden_size_1)
        self.bn5 = nn.BatchNorm1d(hidden_size_1)
        self.fc6 = nn.Linear(hidden_size_1, feature_size)

        self.activation = nn.ReLU()

    def encode(self, x, c):
        hidden = torch.cat([x, c], 1)
        hidden = self.fc1(hidden)
        hidden = self.activation(hidden)
        hidden = self.bn1(hidden)
        hidden = self.fc2(hidden)
        hidden, z1 = self.activation(hidden).split([self.hidden_size_2, self.latent_size], 1)
        hidden = self.bn2(hidden)
        z2 = self.fc3(hidden)

        z1_mu, z1_var = z1.split(self.latent_size // 2, 1)
        z2_mu, z2_var = z2.split(self.latent_size // 2, 1)
        z_mu = torch.cat([z1_mu, z2_mu], 1)
        z_var = torch.cat([z1_var, z2_var], 1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z, c):
        z1, z2 = z.split(self.latent_size // 2, 1)
        hidden = torch.cat([z2, c], 1)
        hidden = self.fc4(hidden)
        hidden = self.activation(hidden)
        hidden = self.bn4(hidden)
        hidden = torch.cat([hidden, z1], 1)
        hidden = self.fc5(hidden)
        hidden = self.activation(hidden)
        hidden = self.bn5(hidden)
        hidden = self.fc6(hidden)
        return hidden

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar


class BaselineCVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size, hidden_size=400, use_batch_norm=False):
        super(BaselineCVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.hidden_size = hidden_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, latent_size)
        self.fc22 = nn.Linear(self.hidden_size, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, feature_size)

        self.relu = nn.ReLU()

        # BatchNorms
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn_fc1 = nn.BatchNorm1d(self.hidden_size)
            self.bn_fc3 = nn.BatchNorm1d(self.hidden_size)

    def encoder_parameters(self):
        modules = [self.fc1, self.fc21, self.fc22]
        if self.use_batch_norm:
            modules.append(self.bn_fc1)
        return [parameter for module in modules for parameter in module.parameters()]

    def decoder_parameters(self):
        modules = [self.fc3, self.fc4]
        if self.use_batch_norm:
            modules.append(self.bn_fc3)
        return [parameter for module in modules for parameter in module.parameters()]

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''

        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        _h1 = self.relu(self.fc1(inputs))
        if self.use_batch_norm:
            h1 = self.bn_fc1(_h1)
        else:
            h1 = _h1
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps*std + mu
        else:
            return mu

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        _h3 = self.relu(self.fc3(inputs))
        if self.use_batch_norm:
            h3 = self.bn_fc3(_h3)
        else:
            h3 = _h3
        out = self.fc4(h3)

        return out

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_size, positions):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('sinusoid_table', self._get_sinusoid_table(hidden_size, positions))

    def _get_denominators(self, hidden_size):
        return torch.tensor([1. / np.power(100000, 2 * (hid_j // 2) / hidden_size) for hid_j in range(hidden_size)]).float()

    def _get_sinusoid_table(self, hidden_size, positions):
        sinusoid_table = positions.unsqueeze(-1).repeat(1, hidden_size) * self._get_denominators(hidden_size)
        sinusoid_table[:, 0::2] = sinusoid_table[:, 0::2].sin()
        sinusoid_table[:, 1::2] = sinusoid_table[:, 1::2].cos()
        return sinusoid_table

    def forward(self, x):
        return x + self.sinusoid_table.clone().detach()


class WindowedTransformer(nn.Module):
    """docstring for WindowedTransformer"""
    def __init__(self, positions: torch.LongTensor, window_size: int, num_output: int, hidden_size: int, num_layers: int, num_classes: int, num_super_classes: int, num_heads=4):
        super(WindowedTransformer, self).__init__()

        self.total_size = positions.shape[0]
        self.window_size = window_size
        class_embedding_size = hidden_size // 4 - 2
        self.class_embedding = nn.Embedding(num_classes + num_super_classes, class_embedding_size)
        positional_embedding_size = hidden_size - class_embedding_size - 2
        self.positional_embedding = nn.Embedding(self.total_size, positional_embedding_size)
        self.positional_encoding = PositionalEncoding(positional_embedding_size, positions)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*2, dropout=0)
        self.transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_size))
        self.output_layer = nn.Linear(hidden_size*2, num_output)

    def forward(self, genotypes, labels, super_labels, maf):
        genotypes = genotypes.unsqueeze(-1)

        maf = maf.unsqueeze(0).unsqueeze(-1).repeat(genotypes.shape[0], 1, 1)

        label_embedding = self.class_embedding(labels).float() + self.class_embedding(super_labels).float()
        label_embedding = label_embedding.unsqueeze(1).repeat(1, genotypes.shape[1], 1)

        position_indices = torch.arange(self.total_size, device=genotypes.device)
        # absolute position embeddings
        positional_embedding = self.positional_embedding(position_indices)
        # add relative position information
        positional_embedding = self.positional_encoding(positional_embedding)
        positional_embedding = positional_embedding.unsqueeze(0).repeat(genotypes.shape[0], 1, 1)
        
        # concat input representation
        x = torch.cat([genotypes, maf, label_embedding, positional_embedding], 2)
        hidden_states = []
        for x_i in x.split(self.window_size, 1):
            hidden_states.append(self.transformer(x_i))
        overlaping_hidden_states = []
        for x_i in (x[:, :self.window_size // 2],) + x[:, self.window_size // 2:].split(self.window_size, 1):
            overlaping_hidden_states.append(self.transformer(x_i))
        hidden_states = torch.cat(hidden_states, 1)
        overlaping_hidden_states = torch.cat(overlaping_hidden_states, 1)
        hidden_states = torch.cat([hidden_states, overlaping_hidden_states], -1)
        logits = self.output_layer(hidden_states)
        return logits

class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, in_size: int, out_size: int, hidden_size: int, num_layers: int):
        super(MLP, self).__init__()
        self.activation = nn.LeakyReLU()
        layers = [nn.Linear(in_size, hidden_size), self.activation]
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.activation)
        layers.append(nn.Linear(hidden_size, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        

class WindowedMLP(nn.Module):
    """docstring for WindowedMLP"""
    def __init__(self, total_size: int, window_size: int, num_layers: int, num_classes: int, num_super_classes: int):
        super(WindowedMLP, self).__init__()

        self.total_size = total_size
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes
        self.mlps = nn.ModuleList([])
        num_models = math.ceil(self.total_size / self.window_size)
        for i in range(num_models):
            if i == num_models - 1:
                window_size = self.total_size % self.window_size
            else:
                window_size = self.window_size
            mlp = MLP(window_size + self.num_classes + self.num_super_classes, window_size, window_size, num_layers)
            self.mlps.append(mlp)

    def forward(self, genotypes, labels, super_labels):

        one_hot_label = nn.functional.one_hot(labels, self.num_classes)
        one_hot_super_label = nn.functional.one_hot(super_labels, self.num_super_classes)
        reconstructed_genotypes = []
        for i, (genotype_i, mlp) in enumerate(zip(genotypes.split(self.window_size, 1), self.mlps)):
            reconstructed_genotype = mlp(torch.cat([genotype_i, one_hot_label, one_hot_super_label], 1))
            reconstructed_genotypes.append(reconstructed_genotype)
        reconstructed_genotypes = torch.cat(reconstructed_genotypes, 1)
        return reconstructed_genotypes


class WindowedModel(nn.Module):
    """docstring for WindowedModel"""
    def __init__(self, ModelClass: Callable, total_size: int, window_size: int, **kwargs):
        super(WindowedModel, self).__init__()

        self.total_size = total_size
        self.window_size = window_size
        self.vaes = nn.ModuleList([])
        self.discriminators = nn.ModuleList([])
        num_models = math.ceil(self.total_size / self.window_size)
        for i in range(num_models):
            if i == num_models - 1:
                new_kwargs = dict([(key, kwargs[key]) for key in kwargs])
                new_kwargs['feature_size'] = self.total_size % self.window_size
                vae = ModelClass(**new_kwargs)
                discriminator = Discriminator(self.total_size % self.window_size, kwargs['class_size'])
            else:
                vae = ModelClass(**kwargs)
                discriminator = Discriminator(self.window_size, kwargs['class_size'])
            self.vaes.append(vae)
            self.discriminators.append(discriminator)

    def decode(self, z, c):
        decoded_xs = []
        for i, (z_i, vae) in enumerate(zip(z.split(z.shape[1] // len(self.vaes), 1), self.vaes)):
            decoded_x = vae.decode(z_i, c)
            if i == len(self.vaes) - 1 and self.total_size % self.window_size != 0:
                decoded_x = decoded_x[:, :self.total_size % self.window_size]
            decoded_xs.append(decoded_x)
        decoded_x = torch.cat(decoded_xs, 1)
        return decoded_x

    def forward(self, x, c):
        reconstructed_xs = []
        mus = []
        logvars = []
        for i, (x_i, vae) in enumerate(zip(x.split(self.window_size, 1), self.vaes)):
            reconstructed_x, mu, logvar = vae(x_i, c)
            if i == len(self.vaes) - 1 and self.total_size % self.window_size != 0:
                reconstructed_x = reconstructed_x[:, :self.total_size % self.window_size]
            reconstructed_xs.append(reconstructed_x)
            mus.append(mu)
            logvars.append(logvar)
        reconstructed_x = torch.cat(reconstructed_xs, 1)
        mu = torch.cat(mus, 1)
        logvar = torch.cat(logvars, 1)
        return reconstructed_x, mu, logvar

    def forward_discriminator(self, x, c):
        outputs = []
        for i, (x_i, discriminator) in enumerate(zip(x.split(self.window_size, 1), self.discriminators)):
            output = discriminator(x_i, c)
            outputs.append(output)
        return torch.cat(outputs, 1)

        # outputs_dis = []

        # for j, dis in enumerate(self.discriminators):
        #     if j == len(self.discriminators)-1:
        #         _x = x[:, j*self.window_size:]
        #     else:
        #         _x = x[:, j * self.window_size : (j + 1) * self.window_size]
        #     out_dis = dis(_x, c)
        #     outputs_dis.append(out_dis)

        # output_dis = torch.stack(outputs_dis,dim=1).squeeze(dim=2)
        # return output_dis


class Discriminator(nn.Module):
    def __init__(self, window_size, num_classes=3):
        super(Discriminator, self).__init__()

        self.hidden_size = 400

        # encode
        input_size = 2 * window_size + num_classes
        # input_size = window_size + num_classes
        self.fc1  = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, c):
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''

        batch = x.shape[0]
        allele_counts = x.mean(0).unsqueeze(0).repeat(batch, 1)
        inputs = torch.cat([x, allele_counts, c], 1)

        # inputs = torch.cat([x, c], 1)

        h1 = self.relu(self.fc1(inputs))
        h2 = self.fc2(h1)
        return h2
