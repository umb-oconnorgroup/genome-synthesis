import math

import torch
from torch import nn


class PopulationLayer(nn.Module):
    """docstring for PopulationLayer"""
    def __init__(self, in_size: int):
        super(PopulationLayer, self).__init__()
        # self.linear = nn.Linear(in_size + 4, in_size)
        self.linear = nn.Linear(in_size, in_size)
        self.population_mean_coef = nn.Parameter(torch.zeros((in_size,)))
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
        population_mean_bias = self.population_mean_coef * population_mean
        if len(x.shape) == 3:
            population_mean_bias = population_mean_bias.unsqueeze(1).repeat(1, x.shape[1], 1)
        return self.linear(x) + population_mean_bias


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
    def __init__(self, in_size: int, num_layers: int):
        super(PopulationMLP, self).__init__()
        layers = [PopulationBlock(in_size // int(math.pow(2, i)), in_size // int(math.pow(2, i + 1))) for i in range(num_layers)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiOutputPopulationNet(nn.Module):
    """docstring for MultiOutputPopulationNet"""
    def __init__(self, in_size: int, out_size: int, num_layers: int):
        super(MultiOutputPopulationNet, self).__init__()
        self.population_mlp = PopulationMLP(in_size, num_layers)
        self.output_layer = nn.Linear(in_size // int(math.pow(2, num_layers)), out_size)

    def forward(self, x):
        return self.output_layer(self.population_mlp(x))


class DiscriminatorCritic(nn.Module):
    """docstring for DiscriminatorCritic"""
    def __init__(self, in_size: int, num_layers: int):
        super(DiscriminatorCritic, self).__init__()
        self.population_mlp = PopulationMLP(in_size, num_layers)
        self.hidden_layer = nn.Linear(in_size // int(math.pow(2, num_layers)), in_size // int(math.pow(2, num_layers)))
        self.discriminator_output_layer = nn.Linear(in_size // int(math.pow(2, num_layers)), 1)
        self.critic_output_layer = nn.Linear(in_size // int(math.pow(2, num_layers)), 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.discriminator_output_layer(self.activation(self.population_mlp(x).mean(-2)))

    def critic_forward(self, x):
        return self.critic_output_layer(self.activation(self.population_mlp(x).mean(-2)))


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
        

class WindowedModel(nn.Module):
    """docstring for WindowedModel"""
    def __init__(self, model_class, total_size: int, window_size: int, num_layers: int, num_classes: int, num_super_classes: int):
        super(WindowedModel, self).__init__()

        self.model_class = model_class
        self.total_size = total_size
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes
        self.models = nn.ModuleList([])
        num_models = math.ceil(self.total_size / self.window_size)
        for i in range(num_models):
            if i == num_models - 1:
                window_size = self.total_size % self.window_size
            else:
                window_size = self.window_size
            if self.model_class == DiscriminatorCritic:
                model = self.model_class(window_size + self.num_classes + self.num_super_classes, num_layers)
            elif self.model_class == MultiOutputPopulationNet:
                model = self.model_class(window_size + self.num_classes + self.num_super_classes, window_size, num_layers)
            else:
                raise NotImplementedError()
            self.models.append(model)
        self.overlaping_models = nn.ModuleList([])
        num_models = math.ceil((self.total_size - self.window_size // 2) / self.window_size) + 1
        for i in range(num_models):
            if i == 0:
                window_size = self.window_size // 2
            elif i == num_models - 1:
                window_size = (self.total_size - self.window_size // 2) % self.window_size
            else:
                window_size = self.window_size
            if self.model_class == DiscriminatorCritic:
                model = self.model_class(window_size + self.num_classes + self.num_super_classes, num_layers)
            elif self.model_class == MultiOutputPopulationNet:
                model = self.model_class(window_size + self.num_classes + self.num_super_classes, window_size, num_layers)
            self.overlaping_models.append(model)

    def forward(self, genotypes, labels, super_labels):

        one_hot_label = nn.functional.one_hot(labels, self.num_classes)
        one_hot_super_label = nn.functional.one_hot(super_labels, self.num_super_classes)

        reconstructed_genotypes = []
        for genotype_i, model in zip(genotypes.split(self.window_size, -1), self.models):
            reconstructed_genotype = model(torch.cat([genotype_i, one_hot_label, one_hot_super_label], -1))
            reconstructed_genotypes.append(reconstructed_genotype)
        reconstructed_genotypes = torch.cat(reconstructed_genotypes, -1)

        reconstructed_overlaping_genotypes = []
        for genotype_i, model in zip((genotypes[..., :self.window_size // 2],) + genotypes[..., self.window_size // 2:].split(self.window_size, -1), self.overlaping_models):
            reconstructed_genotype = model(torch.cat([genotype_i, one_hot_label, one_hot_super_label], -1))
            reconstructed_overlaping_genotypes.append(reconstructed_genotype)
        reconstructed_overlaping_genotypes = torch.cat(reconstructed_overlaping_genotypes, -1)

        if self.model_class == DiscriminatorCritic:
            predictions = torch.cat([reconstructed_genotypes, reconstructed_overlaping_genotypes], -1)
            return predictions
        else:
            reconstructed_genotypes = reconstructed_genotypes + reconstructed_overlaping_genotypes
            return reconstructed_genotypes

    def critic_forward(self, genotypes, labels, super_labels):

        one_hot_label = nn.functional.one_hot(labels, self.num_classes)
        one_hot_super_label = nn.functional.one_hot(super_labels, self.num_super_classes)

        reconstructed_genotypes = []
        for genotype_i, model in zip(genotypes.split(self.window_size, -1), self.models):
            reconstructed_genotype = model.critic_forward(torch.cat([genotype_i, one_hot_label, one_hot_super_label], -1))
            reconstructed_genotypes.append(reconstructed_genotype)
        reconstructed_genotypes = torch.cat(reconstructed_genotypes, -1)

        reconstructed_overlaping_genotypes = []
        for genotype_i, model in zip((genotypes[..., :self.window_size // 2],) + genotypes[..., self.window_size // 2:].split(self.window_size, -1), self.overlaping_models):
            reconstructed_genotype = model.critic_forward(torch.cat([genotype_i, one_hot_label, one_hot_super_label], -1))
            reconstructed_overlaping_genotypes.append(reconstructed_genotype)
        reconstructed_overlaping_genotypes = torch.cat(reconstructed_overlaping_genotypes, -1)

        predictions = torch.cat([reconstructed_genotypes, reconstructed_overlaping_genotypes], -1)
        return predictions
