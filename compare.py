import sys
from typing import List, Tuple

import allel
from matplotlib import pyplot as plt
import numpy as np

from data import biallelic_variant_filter, VCF_FIELDS


def main() -> None:
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]

    callset1 = allel.read_vcf(file_path1, fields=VCF_FIELDS)
    callset2 = allel.read_vcf(file_path2, fields=VCF_FIELDS)

    genotypes1, positions1, _, _ = biallelic_variant_filter(callset1)
    genotypes2, positions2, _, _ = biallelic_variant_filter(callset2)

    shared_position_indices1, shared_position_indices2 = joint_position_indices(positions1, positions2)
    genotypes1 = genotypes1[shared_position_indices1]
    genotypes2 = genotypes2[shared_position_indices2]

    site_frequency_spectrum(genotypes1)
    site_frequency_spectrum(genotypes2)
    joint_site_frequency_spectrum(genotypes1, genotypes2)

def site_frequency_spectrum(genotypes: np.ndarray) -> None:
    allele_counts = genotypes.reshape(genotypes.shape[0], -1).sum(1)
    sfs = allel.sfs(allele_counts, np.product(genotypes.shape[1:]))
    ax = allel.plot_sfs(sfs)
    plt.show()

def joint_position_indices(positions1: np.array, positions2: np.array) -> Tuple[List[int], List[int]]:
    shared_positions = set(positions1).intersection(set(positions2))
    shared_position_indices1 = [i for i in range(len(positions1)) if positions1[i] in shared_positions]
    shared_position_indices2 = [i for i in range(len(positions2)) if positions2[i] in shared_positions]
    return shared_position_indices1, shared_position_indices2

def joint_site_frequency_spectrum(genotypes1: np.ndarray, genotypes2: np.ndarray) -> None:

    allele_counts1 = genotypes1.reshape(genotypes1.shape[0], -1).sum(1)
    allele_counts2 = genotypes2.reshape(genotypes2.shape[0], -1).sum(1)

    joint_sfs = allel.joint_sfs(allele_counts1, allele_counts2, np.product(genotypes1.shape[1:]), np.product(genotypes2.shape[1:]))
    ax = allel.plot_joint_sfs(joint_sfs)
    plt.show()

if __name__ == '__main__':
    main()
