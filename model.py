import math

import torch
from torch import nn


class PopulationLayer(nn.Module):
    """docstring for PopulationLayer"""
    def __init__(self, in_size: int):
        super(PopulationLayer, self).__init__()
        # self.linear = nn.Linear(in_size + 4, in_size)
        self.linear = nn.Linear(in_size, in_size)
        self.population_mean_coef = nn.Parameter(torch.Tensor([in_size]))
        bound = 1. / math.sqrt(in_size)
        nn.init.uniform_(self.population_mean_coef, -bound, bound)

    def normalize(self, x):
        return (x - x.mean()) / x.std()

    def pairwise_distance_distribution(self, x):
        distance_matrix = torch.cdist(x, x)
        distance_mean = distance_matrix.mean(-1)
        diffs = distance_matrix - distance_mean
        distance_var = diffs.pow(2).mean(-1)
        distance_std = distance_var.sqrt()
        zscores = diffs / distance_std
        distance_skews = zscores.pow(3).mean(-1)
        distance_kurtoses = zscores.pow(4).mean(-1)
        return self.normalize(distance_mean), self.normalize(distance_var), self.normalize(distance_skews), self.normalize(distance_kurtoses)

    def forward(self, x):
        population_mean = x.mean(-2)
        # distance_mean, distance_var, distance_skews, distance_kurtoses = self.pairwise_distance_distribution(x)
        # pairwise_distance_augmented_input = torch.cat([x, distance_mean.unsqueeze(-1), distance_var.unsqueeze(-1), distance_skews.unsqueeze(-1), distance_kurtoses.unsqueeze(-1)], -1)
        # return self.linear(pairwise_distance_augmented_input) + self.population_mean_coef * population_mean
        return self.linear(x) + self.population_mean_coef * population_mean


class PopulationBlock(nn.Module):
    """docstring for PopulationBlock"""
    def __init__(self, in_size: int, out_size: int):
        super(PopulationBlock, self).__init__()
        self.activation = nn.LeakyReLU()
        self.population_layer = PopulationLayer(in_size)
        self.linear_layer = nn.Linear(in_size, out_size)

    def forward(self, x):
        h = self.population_layer(x)
        h = self.activation(h)
        h = self.linear_layer(h)
        h = self.activation(h)
        return h

class PopulationMLP(nn.Module):
    """docstring for PopulationMLP"""
    def __init__(self, in_size: int, out_size: int, num_layers: int):
        super(PopulationMLP, self).__init__()
        layers = [PopulationBlock(in_size // int(math.pow(2, i)), in_size // int(math.pow(2, i + 1))) for i in range(num_layers)]
        layers.append(nn.Linear(in_size // int(math.pow(2, num_layers)), out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
        

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
            mlp = PopulationMLP(window_size + self.num_classes + self.num_super_classes, window_size, num_layers)
            # mlp = MLP(window_size + self.num_classes + self.num_super_classes, window_size, window_size // 2, num_layers)
            self.mlps.append(mlp)
        self.overlaping_mlps = nn.ModuleList([])
        num_models = math.ceil((self.total_size - self.window_size // 2) / self.window_size) + 1
        for i in range(num_models):
            if i == 0:
                window_size = self.window_size // 2
            elif i == num_models - 1:
                window_size = (self.total_size - self.window_size // 2) % self.window_size
            else:
                window_size = self.window_size
            mlp = PopulationMLP(window_size + self.num_classes + self.num_super_classes, window_size, num_layers)
            # mlp = MLP(window_size + self.num_classes + self.num_super_classes, window_size, window_size // 2, num_layers)
            self.overlaping_mlps.append(mlp)

    def forward(self, genotypes, labels, super_labels):

        one_hot_label = nn.functional.one_hot(labels, self.num_classes)
        one_hot_super_label = nn.functional.one_hot(super_labels, self.num_super_classes)

        reconstructed_genotypes = []
        for genotype_i, mlp in zip(genotypes.split(self.window_size, 1), self.mlps):
            reconstructed_genotype = mlp(torch.cat([genotype_i, one_hot_label, one_hot_super_label], 1))
            reconstructed_genotypes.append(reconstructed_genotype)
        reconstructed_genotypes = torch.cat(reconstructed_genotypes, 1)

        reconstructed_overlaping_genotypes = []
        for genotype_i, mlp in zip((genotypes[:, :self.window_size // 2],) + genotypes[:, self.window_size // 2:].split(self.window_size, 1), self.overlaping_mlps):
            reconstructed_genotype = mlp(torch.cat([genotype_i, one_hot_label, one_hot_super_label], 1))
            reconstructed_overlaping_genotypes.append(reconstructed_genotype)
        reconstructed_overlaping_genotypes = torch.cat(reconstructed_overlaping_genotypes, 1)

        reconstructed_genotypes = reconstructed_genotypes + reconstructed_overlaping_genotypes
        return reconstructed_genotypes
