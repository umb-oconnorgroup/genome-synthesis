from argparse import ArgumentParser
import math
import os
import random

import torch
import torch.backends.cudnn as cudnn

from loss import corr_coef
from model import WindowedModel
from train import WINDOW_SIZE
from utils import get_device


parser = ArgumentParser(description='Genome Synthesis')
parser.add_argument('-p', '--population', type=str, required=True,
                    help='population code of the samples to be generated')
parser.add_argument('-c', '--checkpoint', type=str, required=True,
                    help='path to the checkpoint that will be used to generate')
parser.add_argument('--data-name', type=str, default=None,
                    help='name added as prefix to file where generated samples will be stored')
parser.add_argument('-d', '--synthetic-dir', type=str, default='synthetic-data',
                    help='path to the directory where the generated samples will be stored')
parser.add_argument('-g', '--gpu', action='store_true',
                    help='use gpu (only supports single gpu)')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='random seed for reproducibility')
parser.add_argument('-n', '--num-samples', type=int, default=100,
                    help='number of diploids to generate')
parser.add_argument('-b', '--batch-size', type=int, default=256,
                    help='training data batch size')
parser.add_argument('--diversity-multiplier', type=int, default=3,
                    help='the most diverse num-samples will be chosen from num-samples multiplied by this number')


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
    if args.population not in label_encoder.classes_:
        raise ValueError('The population of the samples to be generated was not in the training data')

    kwargs = checkpoint['model_kwargs']
    model = WindowedModel(checkpoint['arch'], vcf_writer.snps.shape[0], WINDOW_SIZE, **kwargs)
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    probs = []
    num_haploids = 2 * args.num_samples * args.diversity_multiplier
    num_iterations = math.ceil(num_haploids / args.batch_size)
    label = torch.tensor(label_encoder.transform([args.population]))

    with torch.no_grad():
        model.eval()
        for i in range(num_iterations):
            if i == num_iterations - 1 and num_haploids % args.batch_size != 0:
                batch_size = num_haploids % args.batch_size
            else:
                batch_size = args.batch_size
            z = torch.randn(batch_size, kwargs['latent_size'] * len(model.models)).to(device)
            labels = label.repeat(batch_size).to(device)
            logits = model.decode(z, labels)
            probs.append(logits.sigmoid())

        probs = torch.cat(probs, 0)
        genotypes = torch.bernoulli(probs)
        genotypes = genotypes * 2 - 1

    if args.diversity_multiplier > 1:
        correlation_coefficients = corr_coef(genotypes)
        diverse_indices = correlation_coefficients.mean(1).argsort()[:args.num_samples * 2]
        genotypes = genotypes[diverse_indices]

    genotypes = genotypes.T
    genotypes = genotypes.reshape(genotypes.shape[0], -1, 2)

    samples = ['Sample{}'.format(i) for i in range(args.num_samples)]

    if args.data_name is None:
        file_name = '{}.chr{}.vcf'.format(args.population, vcf_writer.chromosome)
    else:
        file_name = '{}.{}.chr{}.vcf'.format(args.data_name, args.population, vcf_writer.chromosome)
    file_path = os.path.join(args.synthetic_dir, file_name)
    vcf_writer.write_vcf(genotypes, samples, file_path)

if __name__ == '__main__':
    main()
