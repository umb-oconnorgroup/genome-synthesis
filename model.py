import math
from typing import Callable

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
        self.tanh = nn.Tanh()

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
        x = self.tanh(x)
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
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # BatchNorms
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn_fc1 = nn.BatchNorm1d(self.hidden_size)
            self.bn_fc3 = nn.BatchNorm1d(self.hidden_size)


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

        out = self.tanh(out)

        return out

    def forward(self, x, c):
        # one hot encode c
        c = nn.functional.one_hot(c, self.class_size)
        mu, logvar = self.encode(x, c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar

class WindowedModel(nn.Module):
    """docstring for WindowedModel"""
    def __init__(self, ModelClass: Callable, total_size: int, window_size: int, **kwargs):
        super(WindowedModel, self).__init__()
        self.total_size = total_size
        self.window_size = window_size
        self.models = nn.ModuleList([])
        num_models = math.ceil(self.total_size / self.window_size)
        for i in range(num_models):
            if i == num_models - 1:
                new_kwargs = dict([(key, kwargs[key]) for key in kwargs])
                new_kwargs['feature_size'] = self.total_size % self.window_size
                model = ModelClass(**new_kwargs)
            else:
                model = ModelClass(**kwargs)
            self.models.append(model)

    def decode(self, z, c):
        if isinstance(self.models[0], BaselineCVAE):
            c = nn.functional.one_hot(c, self.models[0].class_size)
        decoded_xs = []
        for i, (z_i, model) in enumerate(zip(z.split(z.shape[1] // len(self.models), 1), self.models)):
            decoded_x = model.decode(z_i, c)
            if i == len(self.models) - 1 and self.total_size % self.window_size != 0:
                decoded_x = decoded_x[:, :self.total_size % self.window_size]
            decoded_xs.append(decoded_x)
        decoded_x = torch.cat(decoded_xs, 1)
        return decoded_x

    def forward(self, x, c):
        reconstructed_xs = []
        mus = []
        logvars = []
        for i, (x_i, model) in enumerate(zip(x.split(self.window_size, 1), self.models)):
            reconstructed_x, mu, logvar = model(x_i, c)
            if i == len(self.models) - 1 and self.total_size % self.window_size != 0:
                reconstructed_x = reconstructed_x[:, :self.total_size % self.window_size]
            reconstructed_xs.append(reconstructed_x)
            mus.append(mu)
            logvars.append(logvar)
        reconstructed_x = torch.cat(reconstructed_xs, 1)
        mu = torch.cat(mus, 1)
        logvar = torch.cat(logvars, 1)
        return reconstructed_x, mu, logvar
