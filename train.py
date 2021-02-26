from argparse import ArgumentParser
import os
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from data import BatchByLabelRandomSampler, VCFReader
from model import WindowedModel, MultiOutputPopulationNet, DiscriminatorCritic
from utils import AverageMeter, ProgressMeter, get_device, save_checkpoint


parser = ArgumentParser(description='Genome Synthesis Training')
parser.add_argument('-d', '--train-data', type=str,
                    help='path to training vcf file')
parser.add_argument('-m', '--classification-map', type=str,
                    help='path to map from sample to class')
parser.add_argument('--class-hierarchy', type=str, default=None,
                    help='path to map from class to superclass')
parser.add_argument('-c', '--chromosome', type=int,
                    help='chromosome selected for training')
parser.add_argument('--model-name', type=str, default='train',
                    help='name added as prefix to file where model checkpoint will be saved')
parser.add_argument('--model-dir', type=str, default='models',
                    help='path to directory where model checkpoint will be saved')
parser.add_argument('-r', '--resume_path', type=str,
                    help='path to mlm model from which you would like to resume')
parser.add_argument('-g', '--gpu', type=int, default=-1,
                    help='index of gpu to use (only supports single gpu), -1 indicates cpu')
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='number of training epochs to run')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('-b', '--batch-size', type=int, default=32,
                    help='training data batch size')
parser.add_argument('--passes', type=int, default=10,
                    help='number of passes to generate the data')
parser.add_argument('--discount-rate', default=0.95, type=float,
                    help='discount rate for future rewards')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('-t', '--validation-split', default=0.2, type=float,
                    help='portion of the data used for validation')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency')
parser.add_argument('--save-freq', default=5, type=int,
                    help='save weights every how many epochs')


def main() -> None:

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    start_epoch = 0

    vcf_reader = VCFReader(args.train_data, args.classification_map, args.chromosome, args.class_hierarchy)
    vcf_writer = vcf_reader.get_vcf_writer()
    train_dataset, validation_dataset = vcf_reader.get_datasets(args.validation_split)
    train_sampler = BatchByLabelRandomSampler(args.batch_size, train_dataset.labels)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    if args.validation_split != 0:
        validation_sampler = BatchByLabelRandomSampler(args.batch_size, validation_dataset.labels)
        validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler)

    print("=> loading checkpoint '{}'".format(args.resume_path))
    checkpoint = torch.load(args.resume_path)
    kwargs = checkpoint['model_kwargs']
    mlm_model = WindowedModel(**kwargs)
    mlm_model.to(get_device(args))
    mlm_model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume_path, checkpoint['epoch']))

    discriminator_critic_kwargs = kwargs.copy()
    discriminator_critic_kwargs['model_class'] = DiscriminatorCritic
    discriminator_critic = WindowedModel(**discriminator_critic_kwargs)
    discriminator_critic.to(get_device(args))

    generator_kwargs = kwargs.copy()
    generator_kwargs['model_class'] = MultiOutputPopulationNet
    generator = WindowedModel(**generator_kwargs)
    generator.to(get_device(args))

    # transfer pretrained model
    for discriminator_critic_model, generator_model, mlm_model in zip(list(discriminator_critic.models) + list(discriminator_critic.overlaping_models), list(generator.models) + list(generator.overlaping_models),  list(mlm_model.models) + list(mlm_model.overlaping_models)):
        discriminator_critic_model.population_mlp.load_state_dict(mlm_model.population_mlp.state_dict())
        generator_model.population_mlp.load_state_dict(mlm_model.population_mlp.state_dict())

    discriminator_critic_optimizer = AdamW(discriminator_critic.parameters(), lr=args.learning_rate)
    generator_optimizer = AdamW(generator.parameters(), lr=args.learning_rate)

    for epoch in range(start_epoch, args.epochs + start_epoch):
        train(train_loader, generator, generator_optimizer, discriminator_critic, nn.functional.binary_cross_entropy_with_logits, nn.functional.smooth_l1_loss, discriminator_critic_optimizer, len(vcf_reader.label_encoder.classes_), len(vcf_reader.super_label_encoder.classes_), vcf_reader.maf, epoch, args)

        if epoch % args.save_freq == 0 or epoch == args.epochs + start_epoch - 1:
            if args.validation_split != 0:
                validate(train_loader, generator, discriminator_critic, nn.functional.binary_cross_entropy_with_logits, nn.functional.smooth_l1_loss, len(vcf_reader.label_encoder.classes_), len(vcf_reader.super_label_encoder.classes_), vcf_reader.maf, epoch, args)

            save_checkpoint({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'generator_kwargs': generator_kwargs,
                'generator_optimizer': generator_optimizer.state_dict(),
                'discriminator_critic_state_dict': discriminator_critic.state_dict(),
                'discriminator_critic_kwargs': discriminator_critic_kwargs,
                'discriminator_critic_optimizer': discriminator_critic_optimizer.state_dict(),
                'vcf_writer': vcf_writer,
                'label_encoder': vcf_reader.label_encoder,
                'super_label_encoder': vcf_reader.super_label_encoder,
                'maf': vcf_reader.maf
            }, False, args.chromosome, args.model_name, args.model_dir)


def train(loader: DataLoader,
        generator: nn.Module, generator_optimizer: Optimizer,
        discriminator_critic: nn.Module, discriminator_criterion: Callable, critic_criterion: Callable, discriminator_critic_optimizer: Optimizer,
        num_classes: int, num_super_classes: int, maf: torch.FloatTensor,
        epoch: int, args: ArgumentParser) -> None:
    batch_time = AverageMeter('Time', ':6.3f')
    generator_losses = AverageMeter('G Loss', ':.4e')
    critic_losses = AverageMeter('C Loss', ':.4e')
    discriminator_losses = AverageMeter('D Loss', ':.4e')
    progress = ProgressMeter(len(loader), [batch_time, generator_losses, critic_losses, discriminator_losses], prefix="Epoch: [{}]".format(epoch))

    discriminator_critic.train()
    generator.train()

    device = get_device(args)
    end = time.time()
    for i, (genotypes, labels, super_labels) in enumerate(loader):

        ### Mask for Masked Language Modeling
        mask_num = torch.randint(1, genotypes.shape[1], (1,)).item()
        mask_scores = torch.rand(genotypes.shape[1])
        mask_indices = mask_scores.argsort(descending=True)[:mask_num]
        masked_genotypes = genotypes[:, mask_indices].reshape(-1)
        targets = (masked_genotypes == 1).float().clone().detach()
        genotypes[:, mask_indices] = 0
        maf_vector = maf[labels[0]]

        genotypes = genotypes.to(device)
        masked_genotypes = masked_genotypes.to(device)
        targets = targets.to(device)
        labels = labels.to(device)
        super_labels = super_labels.to(device)
        maf_vector = maf_vector.to(device)

        ### Train
        logits = model(genotypes, labels, super_labels)
        logits = logits[:, mask_indices].reshape(-1)

        # add weight to nonzero maf snps
        weights = torch.ones_like(logits)
        weight_coefficients = (maf_vector[mask_indices] > 0).repeat(genotypes.shape[0]).float() * (args.minor_coefficient - 1) + 1
        weights *= weight_coefficients

        loss = criterion(logits, targets, weight=weights, reduction='mean')
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = (masked_genotypes * logits.sign()).mean() / 2 + .5
        baseline_accuracy = (masked_genotypes * (maf_vector[mask_indices].repeat(genotypes.shape[0]) - .5000001).sign()).mean() / 2 + .5
        accuracy_delta = accuracy - baseline_accuracy

        losses.update(loss.item(), genotypes.shape[0])
        accuracies.update(accuracy.item(), genotypes.shape[0])
        accuracy_deltas.update(accuracy_delta.item(), genotypes.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg



    for i in range(1):
        losses = []
        for j, (genotypes, labels, super_labels) in enumerate(train_loader):

            device = get_device(args)
            maf_vector = maf[labels[0]]

            # add noise to a copy of the population
            noisy_genotypes = genotypes.clone()
            noise_nums = torch.randint(1, genotypes.shape[1], (genotypes.shape[0],))
            noise_scores = torch.rand(genotypes.shape)
            sorted_indices = noise_scores.argsort(descending=True)
            for k, (noise_num, indices) in enumerate(zip(noise_nums, sorted_indices)):
                noise_indices = indices[:noise_num.item()]
                noise = torch.bernoulli(maf_vector[noise_indices]) * 2 - 1
                noisy_genotypes[k][noise_indices] = noise
            real_fake_genotypes = torch.cat([genotypes.unsqueeze(0), noisy_genotypes.unsqueeze(0)], 0)
            targets = torch.tensor([1., 0.])

            noisy_genotypes = noisy_genotypes.to(device)
            real_fake_genotypes = real_fake_genotypes.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            super_labels = super_labels.to(device)
            maf_vector = maf_vector.to(device)

            for j in range(num_passes):
                for logits_by_window in generator(noisy_genotypes, labels, super_labels).split(args.window_size, -1):
                    probabilities = nn.functional.softmax(logits_by_window.reshape(-1), 0)
                    index = torch.multinomial(probabilities, 1).item()
                    logit = logits_by_window[index // logits_by_window.shape[1], index % logits_by_window.shape[1]]
                    noisy_genotypes[index // logits_by_window.shape[1], index % logits_by_window.shape[1]] *= -1
                # get reward from discriminator 

                import sys
                sys.exit(0)

            labels = labels.unsqueeze(0).repeat(2, 1)
            super_labels = super_labels.unsqueeze(0).repeat(2, 1)

            logits = discriminator_critic(real_fake_genotypes, labels, super_labels)
            targets = targets.unsqueeze(1).repeat(1, logits.shape[1])

            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            discriminator_critic.zero_grad()
            loss.backward()
            discriminator_critic_optimizer.step()

            losses.append(loss.item())
        print(np.mean(losses))

    ##########




def validate(loader: DataLoader, model: nn.Module, criterion: Callable, num_classes: int, num_super_classes: int, maf: torch.FloatTensor, args: ArgumentParser) -> torch.FloatTensor:

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('MLM Loss', ':.4e')
    accuracies = AverageMeter('Acc', ':.4e')
    accuracy_deltas = AverageMeter('Acc Delta', ':.4e')
    progress = ProgressMeter(len(loader), [batch_time, losses, accuracies, accuracy_deltas], prefix="Test: ")

    model.eval()

    device = get_device(args)
    with torch.no_grad():
        end = time.time()
        for i, (genotypes, labels, super_labels) in enumerate(loader):

            ### Mask for Masked Language Modeling
            mask_num = int((i % 9 + 1) / 10 * genotypes.shape[1])
            mask_scores = torch.rand(genotypes.shape[1])
            mask_indices = mask_scores.argsort(descending=True)[:mask_num]
            masked_genotypes = genotypes[:, mask_indices].reshape(-1)
            targets = (masked_genotypes == 1).float().clone().detach()
            genotypes[:, mask_indices] = 0
            maf_vector = maf[labels[0]]

            genotypes = genotypes.to(device)
            masked_genotypes = masked_genotypes.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            super_labels = super_labels.to(device)
            maf_vector = maf_vector.to(device)

            logits = model(genotypes, labels, super_labels)
            logits = logits[:, mask_indices].reshape(-1)

            # add weight to nonzero maf snps
            weights = torch.ones_like(logits)
            weight_coefficients = (maf_vector[mask_indices] > 0).repeat(genotypes.shape[0]).float() * (args.minor_coefficient - 1) + 1
            weights *= weight_coefficients

            loss = criterion(logits, targets, weight=weights, reduction='mean')
            
            accuracy = (masked_genotypes * logits.sign()).mean() / 2 + .5
            baseline_accuracy = (masked_genotypes * (maf_vector[mask_indices].repeat(genotypes.shape[0]) - .5000001).sign()).mean() / 2 + .5
            accuracy_delta = accuracy - baseline_accuracy

            losses.update(loss.item(), genotypes.shape[0])
            accuracies.update(accuracy.item(), genotypes.shape[0])
            accuracy_deltas.update(accuracy_delta.item(), genotypes.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display(i)
    return losses.avg


if __name__ == '__main__':
    main()
