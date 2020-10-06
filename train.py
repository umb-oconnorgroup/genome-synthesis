from argparse import ArgumentParser
import os
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from data import VCFReader
from loss import vae_loss as loss_function
from model import BaselineCVAE, CVAE, WindowedModel
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
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='number of training epochs to run')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('-l', '--latent-size', type=int, default=10,
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
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    help='print frequency (default: 10)')

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)

    # model_class = CVAE
    # kwargs = {'latent': args.latent_size, 'num_classes': len(vcf_reader.label_encoder.classes_), 'feature_size': WINDOW_SIZE}
    model_class = BaselineCVAE
    kwargs = {'feature_size': WINDOW_SIZE, 'latent_size': args.latent_size, 'class_size': len(vcf_reader.label_encoder.classes_), 'hidden_size': 100, 'use_batch_norm': True}

    model = WindowedModel(model_class, vcf_reader.snps.shape[0], WINDOW_SIZE, **kwargs)
    model.to(get_device(args))

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    if args.resume_path is not None:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            if not isinstance(model.models[0], checkpoint['arch']):
                raise TypeError('WindowedModel\'s submodels should be instances of {}'.format(checkpoint['arch']))
            if kwargs != checkpoint['kwargs']:
                raise ValueError('The checkpoint\'s kwargs don\'t match the ones used to initialize the model')
            if vcf_reader.snps.shape[0] != checkpoint['vcf_writer'].snps.shape[0]:
                raise ValueError('The data on which the checkpoint was trained had a different number of snp positions')
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.validate:
        validate(validation_loader, model, loss_function, args)
        return

    for epoch in range(start_epoch, args.epochs + start_epoch):
        train(train_loader, model, loss_function, optimizer, epoch, args)
        validation_loss = validate(validation_loader, model, loss_function, args)
        is_best = validation_loss < best_loss
        best_loss = min(validation_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model_class,
            'state_dict': model.state_dict(),
            'model_kwargs': kwargs,
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'vcf_writer': vcf_writer,
            'label_encoder': vcf_reader.label_encoder
        }, is_best, args.chromosome, args.model_name, args.model_dir)

def train(loader: DataLoader, model: torch.nn.Module, criterion: Callable, optimizer: Optimizer, epoch: int, args: ArgumentParser) -> None:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    reconstruction_losses = AverageMeter('Recon Loss', ':.4e')
    kld_losses = AverageMeter('KLD Loss', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, reconstruction_losses, kld_losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    device = get_device(args)
    end = time.time()
    for i, (genotypes, labels) in enumerate(loader):
        genotypes = genotypes.to(device)
        labels = labels.to(device)
        reconstructed_genotypes, mu, logvar = model(genotypes, labels)
        loss, reconstruction, kld = criterion(genotypes, reconstructed_genotypes, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), genotypes.shape[0])
        kld_losses.update(kld.item(), genotypes.shape[0])
        reconstruction_losses.update(reconstruction.item(), genotypes.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(loader: DataLoader, model: torch.nn.Module, criterion: Callable, args: ArgumentParser) -> None:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    reconstruction_losses = AverageMeter('Recon Loss', ':.4e')
    kld_losses = AverageMeter('KLD Loss', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, reconstruction_losses, kld_losses],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (genotypes, labels) in enumerate(loader):
            device = get_device(args)
            genotypes = genotypes.to(device)
            labels = labels.to(device)
            reconstructed_genotypes, mu, logvar = model(genotypes, labels)
            loss, reconstruction, kld = criterion(genotypes, reconstructed_genotypes, mu, logvar)

            losses.update(loss.item(), genotypes.shape[0])
            kld_losses.update(kld.item(), genotypes.shape[0])
            reconstruction_losses.update(reconstruction.item(), genotypes.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    return losses.avg


if __name__ == '__main__':
    main()
