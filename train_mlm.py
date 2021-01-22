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
from model import WindowedMLP
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
parser.add_argument('--model-name', type=str, default='tran',
                    help='name added as prefix to file where model checkpoint will be saved')
parser.add_argument('--model-dir', type=str, default='models',
                    help='path to directory where model checkpoint will be saved')
parser.add_argument('-r', '--resume_path', type=str, default=None,
                    help='path to model from which you would like to resume')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='use gpu (only supports single gpu)')
parser.add_argument('-e', '--epochs', type=int, default=50,
                    help='number of training epochs to run')
# parser.add_argument('--hidden', type=int, default=128,
#                     help='hidden dimension size')
parser.add_argument('-l', '--layers', type=int, default=4,
                    help='number of transformer layers')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('-w', '--window-size', type=int, default=1024,
                    help='size of the window that the snp positions are split into')
parser.add_argument('-b', '--batch-size', type=int, default=32,
                    help='training data batch size')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--minor-coefficient', default=3.0, type=float,
                    help='coefficient for updating weights linked to snps with nonzero maf')
parser.add_argument('-t', '--validation-split', default=0.2, type=float,
                    help='portion of the data used for validation')
parser.add_argument('-v', '--validate', action='store_true',
                    help='only evaluate model on validation set')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency')
parser.add_argument('--save-freq', default=10, type=int,
                    help='save weights every how many epochs')

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

    vcf_reader = VCFReader(args.train_data, args.classification_map, args.chromosome, args.class_hierarchy)
    vcf_writer = vcf_reader.get_vcf_writer()
    train_dataset, validation_dataset = vcf_reader.get_datasets(args.validation_split)
    train_sampler = BatchByLabelRandomSampler(args.batch_size, train_dataset.labels)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    if args.validation_split != 0:
        validation_sampler = BatchByLabelRandomSampler(args.batch_size, validation_dataset.labels)
        validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler)

    kwargs = {'total_size': vcf_reader.positions.shape[0], 'window_size': args.window_size, 'num_layers': args.layers, 'num_classes': len(vcf_reader.label_encoder.classes_), 'num_super_classes': len(vcf_reader.super_label_encoder.classes_)}
    model = WindowedMLP(**kwargs)
    model.to(get_device(args))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    #######
    if args.resume_path is not None:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            if kwargs != checkpoint['model_kwargs']:
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
    #############

    if args.validate:
        validate(validation_loader, model, nn.functional.binary_cross_entropy_with_logits, len(vcf_reader.label_encoder.classes_), len(vcf_reader.super_label_encoder.classes_), vcf_reader.maf, args)
        return

    for epoch in range(start_epoch, args.epochs + start_epoch):
        loss = train(train_loader, model, nn.functional.binary_cross_entropy_with_logits, optimizer, len(vcf_reader.label_encoder.classes_), len(vcf_reader.super_label_encoder.classes_), vcf_reader.maf, epoch, args)

        if epoch % args.save_freq == 0 or epoch == args.epochs + start_epoch - 1:
            if args.validation_split != 0:
                validation_loss = validate(validation_loader, model, nn.functional.binary_cross_entropy_with_logits, len(vcf_reader.label_encoder.classes_), len(vcf_reader.super_label_encoder.classes_), vcf_reader.maf, args)
                is_best = validation_loss < best_loss
                best_loss = min(validation_loss, best_loss)
            else:
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'model_kwargs': kwargs,
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'vcf_writer': vcf_writer,
                'label_encoder': vcf_reader.label_encoder,
                'super_label_encoder': vcf_reader.super_label_encoder,
                'maf': vcf_reader.maf
            }, is_best, args.chromosome, args.model_name, args.model_dir)

def train(loader: DataLoader, model: nn.Module, criterion: Callable, optimizer: Optimizer,
        num_classes: int, num_super_classes: int, maf: torch.FloatTensor,
        epoch: int, args: ArgumentParser) -> torch.FloatTensor:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('MLM Loss', ':.4e')
    accuracies = AverageMeter('Acc', ':.4e')
    accuracy_deltas = AverageMeter('Acc Delta', ':.4e')
    progress = ProgressMeter(len(loader), [batch_time, losses, accuracies, accuracy_deltas], prefix="Epoch: [{}]".format(epoch))

    model.train()

    device = get_device(args)
    end = time.time()
    for i, (genotypes, labels, super_labels) in enumerate(loader):

        ### Mask for Masked Language Modeling
        mask_num = int((torch.distributions.beta.Beta(1.7, 3).sample() * genotypes.shape[1]).round().item())
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
            mask_num = int((torch.distributions.beta.Beta(1.7, 3).sample() * genotypes.shape[1]).round().item())
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
    return losses.avg


if __name__ == '__main__':
    main()
