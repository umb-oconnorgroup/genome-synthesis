from argparse import ArgumentParser
import os
import shutil

import torch


def save_checkpoint(state, is_best, chromosome, name_prefix, dir_path):
    file_name = '{}.chr{}.checkpoint.pth.tar'.format(name_prefix, chromosome)
    file_path = os.path.join(dir_path, file_name)
    torch.save(state, file_path)
    if is_best:
        best_name = '{}.chr{}.best.pth.tar'.format(name_prefix, chromosome)
        best_path = os.path.join(dir_path, best_name)
        shutil.copyfile(file_path, best_path)

def get_device(args: ArgumentParser) -> torch.device:
    if args.gpu >= 0:
        return torch.device('cuda:{}'.format(args.gpu))
    else:
        return torch.device('cpu')

def count_parameters(model: torch.nn.Module) -> int:
    total = 0
    for param in model.parameters():
        total += np.product(param.shape)
    return total


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'