from typing import Tuple

import allel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset


class GenotypeDataset(Dataset):
    """docstring for GenotypeDataset"""
    def __init__(self, vcf_path: str, classification_map_path: str, chromosome: int) -> None:
        super(GenotypeDataset, self).__init__()
        self.chromosome = str(chromosome)
        callset = self.read_vcf(vcf_path)
        samples = callset['samples']
        classification_map = self.read_classification_map(classification_map_path)
        if np.any([not sample in classification_map.index for sample in samples]):
            raise('Some of the samples in the VCF file do not appear in the classification_map')
        classifications = [classification_map.loc[sample]['class'] for sample in samples]
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(classifications)
        genotypes, positions, refs, alts = self.filter_data(callset)
        self.snps = pd.DataFrame(np.stack([refs, alts], axis=1), index=positions, columns=['REF', 'ALT'])
        self.genotypes = self.encode_pos_neg(genotypes)

    def read_vcf(self, file_path: str) -> dict:
        fields = ['calldata/GT', 'samples', 'variants/ALT', 'variants/CHROM', 'variants/FILTER_PASS', 'variants/POS', 'variants/REF', 'variants/is_snp', 'variants/numalt']
        return allel.read_vcf(file_path, fields=fields)

    def read_classification_map(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t', header=None, index_col=0, names=['class'])

    def filter_data(self, callset: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chromosome_filter = callset['variants/CHROM'] == self.chromosome
        vcf_filter = callset['variants/FILTER_PASS']
        snp_filter = callset['variants/is_snp']
        biallelic_filter = callset['variants/numalt'] == 1
        combined_filter = np.all(np.stack([chromosome_filter, vcf_filter, snp_filter, biallelic_filter]), axis=0)
        if combined_filter.sum() < 1:
            raise ValueError('All positions were filtered out in filter_data')
        return callset['calldata/GT'][combined_filter], callset['variants/POS'][combined_filter], callset['variants/REF'][combined_filter], callset['variants/ALT'][combined_filter][:, 0]

    def encode_pos_neg(self, genotype: np.ndarray) -> torch.FloatTensor:
        genotype = torch.tensor(genotype)
        genotype = genotype * 2 - 1
        genotype[genotype == -3] = 0
        return genotype.float()

    def decode_pos_neg(self, genotype: torch.FloatTensor) -> np.ndarray:
        genotype[genotype == 0] = -3
        genotype = .5 * (genotype + 1)
        return genotype.char().numpy()

    def write_vcf(self, genotypes: torch.FloatTensor, file_path: str) -> None:
        # genotypes = self.decode_pos_neg(genotypes)
        pass


vcf_path = 'data/chr20_kgp_abridged.vcf'
classification_map_path = 'data/classification_map.tsv'
chromosome = 20
dataset = GenotypeDataset(vcf_path, classification_map_path, chromosome)
output_file_path = 'synthetic-data/chr{}.vcf'.format(chromosome)
dataset.write_vcf(dataset.genotypes, output_file_path)
