import sys
import os

import allel
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data import biallelic_variant_filter, VCF_FIELDS


SEED = 0
FIGURES_DIR = 'figures'
DATA_DIR = 'data'
CLASSIFICATION_MAP_FILE_NAME = 'classification_map.tsv'
CLASS_HIERARCHY_MAP_FILE_NAME = 'populations.csv'
SUPERPOPULATION_COLORS = {
    'EAS': 'green',
    'EUR': 'blue',
    'AFR': 'red',
    'AMR': 'yellow',
    'SAS': 'purple'
    }
COLOR_PROGRESSION = {
    'green': ['green', 'lawngreen', 'darkturquoise', 'darkolivegreen', 'forestgreen'],
    'blue': ['blue', 'cornflowerblue', 'azure', 'navy', 'lightsteelblue'],
    'red': ['red', 'tomato', 'crimson', 'darkorange', 'firebrick', 'indianred', 'coral'],
    'yellow': ['yellow', 'seashell', 'lightgoldenrodyellow', 'darkkhaki', 'gold'],
    'purple': ['purple', 'lavender', 'darkviolet', 'darkmagenta', 'orchid']
}
SYNTHETIC_COLOR = 'black'
SHAPE_PROGRESSION = ['o', 's', 'D', 'P', 'X', 'p', '*']
SCATTERPLOT_SIZE = 3
SCATTERPLOT_ALPHA = .75
FIGURE_DPI = 500
LEGEND_X_COORDINATES = {
    'EAS': 1.22,
    'EUR': 1.02,
    'AFR': 1.02,
    'AMR': 1.22,
    'SAS': 1.42
}
LEGEND_Y_COORDINATES = {
    'EAS': 0.5,
    'EUR': 0.5,
    'AFR': 1.0,
    'AMR': 1.0,
    'SAS': 0.5
}


def pca(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map, n_components=10):
    pca_algorithm = PCA(n_components=n_components)
    reference_data = reference_genotypes.reshape(reference_genotypes.shape[0], -1).transpose()
    synthetic_data = synthetic_genotypes.reshape(synthetic_genotypes.shape[0], -1).transpose()
    pca_algorithm.fit(np.concatenate((reference_data, synthetic_data), axis=0))
    reference_principle_components = pca_algorithm.transform(reference_data)
    synthetic_principle_components = pca_algorithm.transform(synthetic_data)

    plt.title('PCA')
    plt.xlabel('PC1 ({:.1f}%)'.format(pca_algorithm.explained_variance_ratio_[0] * 100))
    plt.ylabel('PC2 ({:.1f}%)'.format(pca_algorithm.explained_variance_ratio_[1] * 100))

    reference_population_labels = [classification_map.loc[sample]['population'] for sample in reference_samples]
    super_population_groups = class_hierarchy_map.groupby('Super Population Code').groups

    for super_population in super_population_groups:
        handles = []
        for population, color, marker in zip(super_population_groups[super_population], COLOR_PROGRESSION[SUPERPOPULATION_COLORS[super_population]], SHAPE_PROGRESSION):
            indices = [j for i in range(len(reference_population_labels)) if reference_population_labels[i] == population for j in (2 * i, 2 * i + 1)]
            filtered_pc_data = reference_principle_components[indices]
            dots = plt.scatter(filtered_pc_data[:, 0], filtered_pc_data[:, 1], s=SCATTERPLOT_SIZE, c=color, marker=marker, alpha=SCATTERPLOT_ALPHA, label=population)
            handles.append(dots)
            if population == synthetic_population_code:
                dots = plt.scatter(synthetic_principle_components[:, 0], synthetic_principle_components[:, 1], s=SCATTERPLOT_SIZE, c=SYNTHETIC_COLOR, marker=marker, alpha=SCATTERPLOT_ALPHA, label='Synthetic\n{}'.format(synthetic_population_code))
                handles.append(dots)
        # legend = plt.legend(handles=handles, title=super_population, bbox_to_anchor=(1.05, 1.0), loc='upper left')
        legend = plt.legend(handles=handles, title=super_population, bbox_to_anchor=(LEGEND_X_COORDINATES[super_population], LEGEND_Y_COORDINATES[super_population]), loc='upper left')
        plt.gca().add_artist(legend)

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(os.path.join(FIGURES_DIR, '{}.pca.png'.format(synthetic_population_code)), dpi=FIGURE_DPI, bbox_inches='tight')

    return synthetic_principle_components, reference_principle_components


def umap():
    pass

def main() -> None:
    np.random.seed(SEED)

    synthetic_file_path = sys.argv[1]
    reference_file_path = sys.argv[2]
    synthetic_population_code, _ = os.path.split(synthetic_file_path)[-1].split('.', 1)

    classification_map = pd.read_csv(os.path.join(DATA_DIR, CLASSIFICATION_MAP_FILE_NAME), sep='\t', header=None, index_col=0, names=['population'])
    class_hierarchy_map = pd.read_csv(os.path.join(DATA_DIR, CLASS_HIERARCHY_MAP_FILE_NAME), index_col=0)

    synthetic_callset = allel.read_vcf(synthetic_file_path, fields=VCF_FIELDS)
    reference_callset = allel.read_vcf(reference_file_path, fields=VCF_FIELDS)

    synthetic_samples = synthetic_callset[VCF_FIELDS[1]]
    reference_samples = reference_callset[VCF_FIELDS[1]]

    synthetic_genotypes, synthetic_positions, _, _ = biallelic_variant_filter(synthetic_callset)
    reference_genotypes, reference_positions, _, _ = biallelic_variant_filter(reference_callset)

    synthetic_principle_components, reference_principle_components = pca(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map)


if __name__ == '__main__':
    main()
