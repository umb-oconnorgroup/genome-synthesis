import os
import sys
from typing import List, Tuple

import allel
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch

from data import biallelic_variant_filter, VCF_FIELDS
from loss import linkage_disequilibrium_correlation


FIGURES_DIR = 'figures'


def main() -> None:
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]

    if len(sys.argv) == 5:
        population1 = sys.argv[3]
        population2 = sys.argv[4]

    callset1 = allel.read_vcf(file_path1, fields=VCF_FIELDS)
    callset2 = allel.read_vcf(file_path2, fields=VCF_FIELDS)

    genotypes1, positions1, _, _ = biallelic_variant_filter(callset1)
    genotypes2, positions2, _, _ = biallelic_variant_filter(callset2)

    shared_position_indices1, shared_position_indices2 = joint_position_indices(positions1, positions2)
    positions1 = positions1[shared_position_indices1]
    positions2 = positions2[shared_position_indices2]
    genotypes1 = genotypes1[shared_position_indices1]
    genotypes2 = genotypes2[shared_position_indices2]

    genotypes1[genotypes1 < 0] = 0
    genotypes2[genotypes2 < 0] = 0

    sfs1 = site_frequency_spectrum(genotypes1, population1)
    sfs2 = site_frequency_spectrum(genotypes2, population2)
    joint_site_frequency_spectrum(genotypes1, genotypes2, population1, population2)

    # linkage_disequilibrium(positions1, genotypes1)
    # linkage_disequilibrium(positions2, genotypes2)

def site_frequency_spectrum(genotypes: np.ndarray, population: str=None) -> np.ndarray:
    allele_counts = genotypes.reshape(genotypes.shape[0], -1).sum(1)
    sfs = allel.sfs(allele_counts, np.product(genotypes.shape[1:]))
    if population is not None:
        plt.title('{} site frequency spectrum'.format(population))
    ax = plt.gca()
    ax = allel.plot_sfs(sfs, ax=ax)
    plt.savefig(os.path.join(FIGURES_DIR, '{}.sfs.png'.format(population.replace(' ', '_'))))
    plt.clf()
    return sfs / sfs.sum()

def joint_position_indices(positions1: np.array, positions2: np.array) -> Tuple[List[int], List[int]]:
    shared_positions = set(positions1).intersection(set(positions2))
    shared_position_indices1 = [i for i in range(len(positions1)) if positions1[i] in shared_positions]
    shared_position_indices2 = [i for i in range(len(positions2)) if positions2[i] in shared_positions]
    return shared_position_indices1, shared_position_indices2

def joint_site_frequency_spectrum(genotypes1: np.ndarray, genotypes2: np.ndarray, population1: str='population1', population2: str='population2') -> np.ndarray:
    allele_counts1 = genotypes1.reshape(genotypes1.shape[0], -1).sum(1)
    allele_counts2 = genotypes2.reshape(genotypes2.shape[0], -1).sum(1)
    joint_sfs = allel.joint_sfs(allele_counts1, allele_counts2, np.product(genotypes1.shape[1:]), np.product(genotypes2.shape[1:]))
    ax = plot_joint_sfs(joint_sfs, population1, population2)
    plt.savefig(os.path.join(FIGURES_DIR, '{}.{}.joint_sfs.png'.format(population1.replace(' ', '_'), population2.replace(' ', '_'))))
    plt.clf()
    return joint_sfs / joint_sfs.sum()

def linkage_disequilibrium(positions: np.array, genotypes: np.ndarray, window_size: int=200000) -> None:
    genotypes = torch.FloatTensor(genotypes.reshape(genotypes.shape[0], -1)).T
    windows = []
    position = 0
    while(position < positions[-1]):
        position_indices = np.where(np.logical_and(positions > position, positions <= position + window_size))[0]
        position += window_size
        if len(position_indices) == 0:
            continue
        r_squared = linkage_disequilibrium_correlation(genotypes[:, position_indices])
        non_nan_rsquared = r_squared[~r_squared.isnan()]
        summary = {
            'count': len(position_indices),
            'r_squared mean': non_nan_rsquared.mean(),
            'r_squared variance': non_nan_rsquared.var()
        }
        windows.append(summary)
    print(windows)

def plot_joint_sfs(s: np.ndarray, population1: str='population1', population2: str='population2') -> matplotlib.axes.Axes:
    """Plot a joint site frequency spectrum.

    Parameters
    ----------
    s : array_like, int, shape (n_chromosomes_pop1, n_chromosomes_pop2)
        Joint site frequency spectrum.
    ax : axes, optional
        Axes on which to draw. If not provided, a new figure will be created.
    imshow_kwargs : dict-like
        Additional keyword arguments, passed through to ax.imshow().

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # check inputs
    s = allel.util.asarray_ndim(s, 2)

    # setup axes
    w = plt.rcParams['figure.figsize'][0]
    fig, ax = plt.subplots(figsize=(w, w))

    # set plotting defaults
    imshow_kwargs = dict()
    imshow_kwargs.setdefault('cmap', 'jet')
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('aspect', 'auto')
    imshow_kwargs.setdefault('norm', LogNorm())

    # plot data
    ax.imshow(s.T, **imshow_kwargs)

    # tidy
    ax.invert_yaxis()
    ax.set_title('joint site frequency spectrum')
    ax.set_xlabel('derived allele count ({})'.format(population1))
    ax.set_ylabel('derived allele count ({})'.format(population2))

    return ax


if __name__ == '__main__':
    main()
