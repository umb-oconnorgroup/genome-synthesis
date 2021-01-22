from argparse import ArgumentParser
import math
import os
import random

import torch
import torch.backends.cudnn as cudnn

from loss import squared_corr_coef
from model import WindowedTransformer
from utils import get_device


parser = ArgumentParser(description='Genome Synthesis')
parser.add_argument('-p', '--population', type=str, required=True,
                    help='population code of the samples to be generated')
parser.add_argument('-s', '--super-population', type=str, required=True,
                    help='super population code of the samples to be generated')
parser.add_argument('-c', '--checkpoint', type=str, required=True,
                    help='path to the checkpoint that will be used to generate')
parser.add_argument('--data-name', type=str, default=None,
                    help='name added as prefix to file where generated samples will be stored')
parser.add_argument('-d', '--synthetic-dir', type=str, default='synthetic-data',
                    help='path to the directory where the generated samples will be stored')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='use gpu (only supports single gpu)')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('-n', '--num-samples', type=int, default=100,
                    help='number of diploids to generate')
parser.add_argument('-b', '--batch-size', type=int, default=8,
                    help='training data batch size')
parser.add_argument('--passes', type=int, default=50,
                    help='training data batch size')
parser.add_argument('--diversity-multiplier', type=int, default=3,
                    help='the most diverse num-samples will be chosen from num-samples multiplied by this number')


def generate(num_passes, model, label, super_label, maf, batch_size, device):
    genotypes = torch.zeros(batch_size, model.total_size).to(device)
    labels = label.repeat(batch_size).to(device)
    super_labels = super_label.repeat(batch_size).to(device)
    maf = maf.to(device)
    for j in range(num_passes):
        num_selected = genotypes.shape[1] // num_passes
        if j == num_passes - 1:
            num_selected += genotypes.shape[1] % num_passes
        logits = model(genotypes, labels, super_labels, maf).squeeze(-1)
        # zero out logits of already selected indices
        logits[genotypes != 0] = 0
        selected_indices = logits.abs().argsort(descending=True)[:, :num_selected]
        for k, selected_idx in enumerate(selected_indices):
            genotypes[k, selected_idx] = logits[k, selected_idx].sign()
    return genotypes


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

    device = get_device(args)

    if not os.path.isfile(args.checkpoint):
        raise ValueError('No checkpoint found at {}'.format(args.checkpoint))

    checkpoint = torch.load(args.checkpoint)

    vcf_writer = checkpoint['vcf_writer']
    label_encoder = checkpoint['label_encoder']
    super_label_encoder = checkpoint['super_label_encoder']
    maf = checkpoint['maf']
    if args.population not in label_encoder.classes_:
        raise ValueError('The population of the samples to be generated was not in the training data')
    if args.super_population not in super_label_encoder.classes_:
        raise ValueError('The super population of the samples to be generated was not in the training data')

    kwargs = checkpoint['model_kwargs']
    model = WindowedTransformer(**kwargs)
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    genotypes = []
    num_haploids = 2 * args.num_samples * args.diversity_multiplier
    num_iterations = math.ceil(num_haploids / args.batch_size)
    label = torch.tensor(label_encoder.transform([args.population]))
    super_label = torch.tensor(super_label_encoder.transform([args.super_population]))
    maf = maf[label[0]]

    with torch.no_grad():
        model.eval()
        for i in range(num_iterations):
            if i == num_iterations - 1 and num_haploids % args.batch_size != 0:
                batch_size = num_haploids % args.batch_size
            else:
                batch_size = args.batch_size
            genotypes.append(generate(args.passes, model, label, super_label, maf, batch_size, device))

        genotypes = torch.cat(genotypes, 0)

        if args.diversity_multiplier > 1:
            squared_correlation_coefficients = squared_corr_coef(genotypes)
            diverse_indices = squared_correlation_coefficients.mean(1).argsort()[:args.num_samples * 2]
            genotypes = genotypes[diverse_indices]

    genotypes = genotypes.T
    genotypes = genotypes.reshape(genotypes.shape[0], -1, 2).cpu()

    samples = ['{}{}'.format(args.population, i) for i in range(args.num_samples)]

    if args.data_name is None:
        file_name = '{}.chr{}.vcf'.format(args.population, vcf_writer.chromosome)
    else:
        file_name = '{}.{}.chr{}.vcf'.format(args.data_name, args.population, vcf_writer.chromosome)
    file_path = os.path.join(args.synthetic_dir, file_name)
    vcf_writer.write_vcf(genotypes, samples, file_path)

if __name__ == '__main__':
    main()
