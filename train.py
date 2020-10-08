from argparse import ArgumentParser
import os
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import Adam, Optimizer, RMSprop
from torch.utils.data import DataLoader

from data import BatchByLabelRandomSampler, VCFReader
from loss import reconstruction_loss, kld_loss
from model import BaselineCVAE, CHVAE, CVAE, WindowedModel
from utils import AverageMeter, ProgressMeter, get_device, save_checkpoint


# Ideally the window_size would be an argument, but one of the model architectures is dependent on
# this size, so until that is changed, it has to be a constant
WINDOW_SIZE = 2048

parser = ArgumentParser(description='Genome Synthesis Training')
parser.add_argument('-d', '--train-data', type=str,
                    help='path to training vcf file')
parser.add_argument('-m', '--classification-map', type=str,
                    help='path to map from sample to class')
parser.add_argument('-c', '--chromosome', type=int,
                    help='chromosome selected for training')
parser.add_argument('--model-name', type=str, default='model',
                    help='name added as prefix to file where model checkpoint will be saved')
parser.add_argument('--model-dir', type=str, default='models',
                    help='path to directory where model checkpoint will be saved')
parser.add_argument('-r', '--resume_path', type=str, default=None,
                    help='path to model from which you would like to resume')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='use gpu (only supports single gpu)')
parser.add_argument('-e', '--epochs', type=int, default=20,
                    help='number of training epochs to run')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('-l', '--latent-size', type=int, default=20,
                    help='size of the latent space for each window')
# parser.add_argument('-w', '--window-size', type=int, default=2048,
                    # help='size of the window that the snp positions are split into')
parser.add_argument('-b', '--batch-size', type=int, default=128,
                    help='training data batch size')
parser.add_argument('--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('-t', '--validation-split', default=0.2, type=float,
                    help='portion of the data used for validation')
parser.add_argument('-v', '--validate', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency')

best_loss = np.inf

def main() -> None:
    global best_loss

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

    vcf_reader = VCFReader(args.train_data, args.classification_map, args.chromosome)
    vcf_writer = vcf_reader.get_vcf_writer()
    train_dataset, validation_dataset = vcf_reader.get_datasets(args.validation_split)
    train_sampler = BatchByLabelRandomSampler(args.batch_size, train_dataset.labels)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    if args.validation_split != 0:
        validation_sampler = BatchByLabelRandomSampler(args.batch_size, validation_dataset.labels)
        validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler)

    # _average_haploid, _std_haploid = calculate_average_haploid(train_loader, args)

    # model_class = CVAE
    # kwargs = {'latent': args.latent_size, 'num_classes': len(vcf_reader.label_encoder.classes_), 'feature_size': WINDOW_SIZE}
    model_class = BaselineCVAE
    kwargs = {'feature_size': WINDOW_SIZE, 'latent_size': args.latent_size, 'class_size': len(vcf_reader.label_encoder.classes_), 'hidden_size': 200, 'use_batch_norm': False}
    # model_class = CHVAE
    # kwargs = {'feature_size': WINDOW_SIZE, 'latent_size': args.latent_size, 'class_size': len(vcf_reader.label_encoder.classes_), 'hidden_size_1': 400, 'hidden_size_2': 200}

    model = WindowedModel(model_class, vcf_reader.snps.shape[0], WINDOW_SIZE, **kwargs)
    model.to(get_device(args))

    encoder_optimizer = RMSprop([parameter for vae in model.vaes for parameter in vae.encoder_parameters()], lr=args.learning_rate)
    decoder_optimizer = RMSprop([parameter for vae in model.vaes for parameter in vae.decoder_parameters()], lr=args.learning_rate)
    discriminator_optimizer = RMSprop(model.discriminators.parameters(), lr=args.learning_rate)

    if args.resume_path is not None:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            if not isinstance(model.vaes[0], checkpoint['arch']):
                raise TypeError('WindowedModel\'s submodels should be instances of {}'.format(checkpoint['arch']))
            if kwargs != checkpoint['kwargs']:
                raise ValueError('The checkpoint\'s kwargs don\'t match the ones used to initialize the model')
            if vcf_reader.snps.shape[0] != checkpoint['vcf_writer'].snps.shape[0]:
                raise ValueError('The data on which the checkpoint was trained had a different number of snp positions')
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.validate:
        validate(validation_loader, model, loss_function, args)
        return

    for epoch in range(start_epoch, args.epochs + start_epoch):
        loss = train(train_loader, model, reconstruction_loss, kld_loss, encoder_optimizer, decoder_optimizer, discriminator_optimizer, len(vcf_reader.label_encoder.classes_), epoch, args)
        # if args.validation_split != 0:
            # validation_loss = validate(validation_loader, model, loss_function, args)
            # is_best = validation_loss < best_loss
            # best_loss = min(validation_loss, best_loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_class,
            'state_dict': model.state_dict(),
            'model_kwargs': kwargs,
            'best_loss': best_loss,
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
            'vcf_writer': vcf_writer,
            'label_encoder': vcf_reader.label_encoder
        }, is_best, args.chromosome, args.model_name, args.model_dir)

def train(loader: DataLoader, model: nn.Module,
        reconstruction_criterion: Callable, kld_criterion: Callable,
        encoder_optimizer: Optimizer, decoder_optimizer: Optimizer, discriminator_optimizer: Optimizer,
        num_classes: int, epoch: int, args: ArgumentParser) -> torch.FloatTensor:
    batch_time = AverageMeter('Time', ':6.3f')
    decoder_losses = AverageMeter('Decoder', ':.4e')
    reconstruction_losses = AverageMeter('Recon', ':.4e')
    kld_losses = AverageMeter('KLD', ':.4e')
    simulation_losses = AverageMeter('Sim', ':.4e')
    discriminator_losses = AverageMeter('Dis', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, decoder_losses, reconstruction_losses, kld_losses, simulation_losses, discriminator_losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    device = get_device(args)
    end = time.time()
    for i, (genotypes, labels) in enumerate(loader):
        genotypes = genotypes.to(device)
        labels = labels.to(device)

        ### Train Discriminator

        n_critic = 5
        k = 4
        alpha = 1
        clip_value = .01
        for j in range(n_critic):
            discriminator_labels = torch.randint(0, num_classes, (1,)).repeat(labels.shape[0]).to(device)
            one_hot_labels = nn.functional.one_hot(discriminator_labels, num_classes)
            z = torch.randn(labels.shape[0], args.latent_size * len(model.vaes)).to(device)
            logits = model.decode(z, one_hot_labels).detach()
            samples = torch.bernoulli(logits.sigmoid())
            fakes = (k * logits).sigmoid() * 2 - 1
            unlikely_sampling_mask = (logits.sign() * (samples - .5).sign()) == -1
            fakes[unlikely_sampling_mask] = -fakes[unlikely_sampling_mask].detach().clone()
            discriminator_out_fake = model.forward_discriminator(fakes, one_hot_labels)
            discriminator_out_real = model.forward_discriminator(genotypes, one_hot_labels)

            discriminator_loss = discriminator_out_fake.sum(1).mean() - discriminator_out_real.sum(1).mean()
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            for param in model.discriminators.parameters():
                param.data.clamp_(-clip_value, clip_value)

        ### Train VAE

        one_hot_labels = nn.functional.one_hot(labels, num_classes)
        logits, mu, logvar = model(genotypes, one_hot_labels)
        reconstruction = reconstruction_criterion(genotypes, logits)
        kld = kld_criterion(mu, logvar)

        encoder_loss = reconstruction + kld
        model.vaes.zero_grad()
        encoder_loss.backward()
        encoder_optimizer.step()

        logits = model.decode(mu.detach().clone(), one_hot_labels)
        reconstruction = reconstruction_criterion(genotypes, logits)
        samples = torch.bernoulli(logits.sigmoid())
        fakes = (k * logits).sigmoid() * 2 - 1
        unlikely_sampling_mask = (logits.sign() * (samples - .5).sign()) == -1
        fakes[unlikely_sampling_mask] = -fakes[unlikely_sampling_mask].detach().clone()
        discriminator_out_fake = model.forward_discriminator(fakes, one_hot_labels)
        simulation_loss = -discriminator_out_fake.sum(1).mean()

        decoder_loss = reconstruction + alpha * simulation_loss
        model.vaes.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()

        decoder_losses.update(decoder_loss.item(), genotypes.shape[0])
        kld_losses.update(kld.item(), genotypes.shape[0])
        reconstruction_losses.update(reconstruction.item(), genotypes.shape[0])
        simulation_losses.update(simulation_loss.item(), genotypes.shape[0])
        discriminator_losses.update(discriminator_loss.item(), genotypes.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return decoder_losses.avg

# def validate(loader: DataLoader, model: nn.Module, vae_criterion: Callable, args: ArgumentParser) -> torch.FloatTensor:
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     reconstruction_losses = AverageMeter('Recon Loss', ':.4e')
#     kld_losses = AverageMeter('KLD Loss', ':.4e')
#     progress = ProgressMeter(
#         len(loader),
#         [batch_time, losses, reconstruction_losses, kld_losses],
#         prefix='Test: ')

#     model.eval()
#     with torch.no_grad():
#         end = time.time()
#         for i, (genotypes, labels) in enumerate(loader):
#             device = get_device(args)
#             genotypes = genotypes.to(device)
#             labels = labels.to(device)
#             logits, mu, logvar = model(genotypes, labels)
#             loss, reconstruction, kld = vae_criterion(genotypes, logits, mu, logvar, WINDOW_SIZE)

#             losses.update(loss.item(), genotypes.shape[0])
#             kld_losses.update(kld.item(), genotypes.shape[0])
#             reconstruction_losses.update(reconstruction.item(), genotypes.shape[0])
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if i % args.print_freq == 0:
#                 progress.display(i)
#     return losses.avg

def calculate_average_haploid(loader: DataLoader, args: ArgumentParser):
    device = get_device(args)
    average_haploid = None
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device)  # .float()
        if average_haploid is None:
            average_haploid = torch.mean(inputs, dim=0)
        else:
            average_haploid += torch.mean(inputs, dim=0)

    _average_haploid = average_haploid / i
    average_haploid = average_haploid.sign()


    std_haploid = None
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device).float(), labels.to(device)  # .float()
        if std_haploid is None:
            std_haploid = torch.mean((inputs - _average_haploid)**2, dim=0)
        else:
            std_haploid += torch.mean((inputs - _average_haploid)**2, dim=0)

    _std_haploid = torch.sqrt(std_haploid / i)
    return _average_haploid, _std_haploid


if __name__ == '__main__':
    main()
