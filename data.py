import math
from typing import List, Tuple

import allel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, Sampler

from vcf_write import write_vcf


VCF_FIELDS = ['calldata/GT', 'samples', 'variants/ALT', 'variants/CHROM', 'variants/FILTER_PASS', 'variants/POS', 'variants/REF', 'variants/is_snp', 'variants/numalt']


class GenotypeDataset(Dataset):
    """docstring for GenotypeDataset"""
    def __init__(self, genotypes: torch.FloatTensor, labels: torch.FloatTensor) -> None:
        super(GenotypeDataset, self).__init__()
        self.genotypes = genotypes
        self.labels = labels

    def __len__(self) -> int:
        return self.genotypes.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.genotypes[index], self.labels[index]


class BatchByLabelRandomSampler(Sampler):
    """docstring for BatchByLabelRandomSampler"""
    def __init__(self, batch_size: int, labels: torch.LongTensor) -> None:
        self.batch_size = batch_size
        self.labels = labels

    def __iter__(self):
        used_label_indices = torch.LongTensor([])
        while len(used_label_indices) < len(self.labels):
            remaining_filter = torch.ones_like(self.labels)
            remaining_filter[used_label_indices] = 0
            remaining_filter = remaining_filter.bool()
            remaining_label_indices = torch.arange(len(self.labels))[remaining_filter]
            label_idx = remaining_label_indices[torch.randint(0, len(remaining_label_indices), (1,))]
            label_indices = torch.where((self.labels == self.labels[label_idx]).logical_and(remaining_filter))[0]
            shuffled_label_indices = label_indices[torch.randperm(len(label_indices))]
            selected_label_indices = shuffled_label_indices[:self.batch_size]
            used_label_indices = torch.cat([used_label_indices, selected_label_indices])
            yield selected_label_indices.tolist()

    def __len__(self):
        return ((torch.bincount(self.labels) + self.batch_size - 1) // self.batch_size).sum()


class VCFWriter(object):
    """docstring for VCFWriter"""
    def __init__(self, chromosome: int, snps: pd.DataFrame) -> None:
        super(VCFWriter, self).__init__()
        self.chromosome = chromosome
        self.snps = snps

    def decode_pos_neg(self, genotype: torch.FloatTensor) -> np.ndarray:
        genotype[genotype == 0] = -3
        genotype = .5 * (genotype + 1)
        return genotype.char().numpy()

    def write_vcf(self, genotypes: torch.FloatTensor, samples: List[str], file_path: str) -> None:
        genotypes = self.decode_pos_neg(genotypes)
        callset = {
            'calldata/GT': genotypes,
            'samples': samples,
            'variants/CHROM': np.repeat(self.chromosome, genotypes.shape[0]),
            'variants/POS': self.snps.index.to_numpy(),
            'variants/ID': np.array(['.' for i in range(genotypes.shape[0])], dtype=object),
            'variants/REF': self.snps['REF'].to_numpy(),
            'variants/ALT': self.snps['ALT'].to_numpy(),
            'variants/QUAL': np.repeat(np.nan, genotypes.shape[0]).astype(np.float32),
            'variants/FILTER_PASS': np.repeat(True, genotypes.shape[0])
        }
        write_vcf(file_path, callset)


class VCFReader(object):
    """docstring for VCFReader"""
    def __init__(self, vcf_path: str, classification_map_path: str, chromosome: int) -> None:
        super(VCFReader, self).__init__()
        self.chromosome = str(chromosome)
        callset = self.read_vcf(vcf_path)
        samples = callset['samples']
        classification_map = self.read_classification_map(classification_map_path)
        if np.any([not sample in classification_map.index for sample in samples]):
            raise('Some of the samples in the VCF file do not appear in the classification_map')
        classifications = [classification_map.loc[sample]['class'] for sample in samples]
        self.label_encoder = LabelEncoder()
        self.labels = torch.tensor(self.label_encoder.fit_transform(classifications))
        genotypes, positions, refs, alts = biallelic_variant_filter(callset, self.chromosome)
        self.snps = pd.DataFrame(np.stack([refs, alts], axis=1), index=positions, columns=['REF', 'ALT'])
        self.genotypes = self.encode_pos_neg(genotypes)
        # transform diploid data into haploid data and apply same transformation to labels
        if len(self.genotypes.shape) == 3:
            self.labels = self.labels.unsqueeze(1).repeat(1, self.genotypes.shape[2]).reshape(-1)
            self.genotypes = self.genotypes.reshape(self.genotypes.shape[0], -1)

    def read_vcf(self, file_path: str) -> dict:
        return allel.read_vcf(file_path, fields=VCF_FIELDS)

    def read_classification_map(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t', header=None, index_col=0, names=['class'])

    def encode_pos_neg(self, genotype: np.ndarray) -> torch.FloatTensor:
        genotype = torch.tensor(genotype)
        genotype = genotype * 2 - 1
        genotype[genotype == -3] = 0
        return genotype.float()

    def get_datasets(self, val_split: float) -> Tuple[GenotypeDataset, GenotypeDataset]:
        if val_split > 1 or val_split < 0:
            raise ValueError('val_split must be in between 0 and 1')
        elif val_split == 0:
            return GenotypeDataset(self.genotypes.T , self.labels), None
        val_size = math.floor(self.genotypes.shape[1] * val_split)
        permutation_idx = torch.randperm(self.genotypes.shape[1])
        permuted_genotypes = self.genotypes.T[permutation_idx]
        permuted_labels = self.labels[permutation_idx]
        return GenotypeDataset(permuted_genotypes[val_size:] , permuted_labels[val_size:]), GenotypeDataset(permuted_genotypes[0: val_size] , permuted_labels[0 :val_size])

    def get_vcf_writer(self) -> VCFWriter:
        return VCFWriter(self.chromosome, self.snps)

def biallelic_variant_filter(callset: dict, chromosome: int=None) -> Tuple[np.ndarray, np.array, np.array, np.array]:
    vcf_filter = callset['variants/FILTER_PASS']
    snp_filter = callset['variants/is_snp']
    biallelic_filter = callset['variants/numalt'] == 1
    if chromosome is None:
        chromosome_filter = np.repeat(np.array([True]), len(biallelic_filter))
    else:
        chromosome_filter = callset['variants/CHROM'] == chromosome
    combined_filter = np.all(np.stack([chromosome_filter, vcf_filter, snp_filter, biallelic_filter]), axis=0)
    if combined_filter.sum() < 1:
        raise ValueError('All positions were filtered out in filter_data')
    return callset['calldata/GT'][combined_filter], callset['variants/POS'][combined_filter], callset['variants/REF'][combined_filter], callset['variants/ALT'][combined_filter][:, 0]
