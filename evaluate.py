import sys
import os

import allel
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

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
    'green': ['green', 'lawngreen', 'springgreen', 'darkolivegreen', 'forestgreen'],
    'blue': ['blue', 'cornflowerblue', 'darkturquoise', 'navy', 'lightsteelblue'],
    'red': ['red', 'tomato', 'crimson', 'darkorange', 'firebrick', 'indianred', 'coral'],
    'yellow': ['goldenrod', 'peru', 'tan', 'darkkhaki', 'gold'],
    'purple': ['purple', 'fuchsia', 'darkviolet', 'hotpink', 'orchid']
}
SYNTHETIC_COLOR = 'black'
SHAPE_PROGRESSION = ['o', 's', 'D', 'p', '*', 'P', 'X']
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


def scatterplot(synthetic_population_code, synthetic_data, reference_data, reference_samples, classification_map, class_hierarchy_map, title, xlabel, ylabel, fig_path):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    reference_population_labels = [classification_map.loc[sample]['population'] for sample in reference_samples]
    super_population_groups = class_hierarchy_map.groupby('Super Population Code').groups

    for super_population in super_population_groups:
        handles = []
        for population, color, marker in zip(super_population_groups[super_population], COLOR_PROGRESSION[SUPERPOPULATION_COLORS[super_population]], SHAPE_PROGRESSION):
            indices = [j for i in range(len(reference_population_labels)) if reference_population_labels[i] == population for j in (2 * i, 2 * i + 1)]
            filtered_reference_data = reference_data[indices]
            dots = plt.scatter(filtered_reference_data[:, 0], filtered_reference_data[:, 1], s=SCATTERPLOT_SIZE, c=color, marker=marker, alpha=SCATTERPLOT_ALPHA, label=population)
            handles.append(dots)
            if population == synthetic_population_code:
                dots = plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], s=SCATTERPLOT_SIZE, c=SYNTHETIC_COLOR, marker=marker, alpha=SCATTERPLOT_ALPHA, label='Synthetic\n{}'.format(synthetic_population_code))
                handles.append(dots)
        legend = plt.legend(handles=handles, markerscale=3, title=super_population, bbox_to_anchor=(LEGEND_X_COORDINATES[super_population], LEGEND_Y_COORDINATES[super_population]), loc='upper left')
        plt.gca().add_artist(legend)

    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(plt.gcf())


def run_pca(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map, n_components=10):
    pca_algorithm = PCA(n_components=n_components)
    reference_data = reference_genotypes.reshape(reference_genotypes.shape[0], -1).transpose()
    synthetic_data = synthetic_genotypes.reshape(synthetic_genotypes.shape[0], -1).transpose()
    pca_algorithm.fit(np.concatenate((reference_data, synthetic_data), axis=0))
    reference_principle_components = pca_algorithm.transform(reference_data)
    synthetic_principle_components = pca_algorithm.transform(synthetic_data)

    title = 'PCA'
    xlabel = 'PC1 ({:.1f}%)'.format(pca_algorithm.explained_variance_ratio_[0] * 100)
    ylabel = 'PC2 ({:.1f}%)'.format(pca_algorithm.explained_variance_ratio_[1] * 100)
    fig_path = os.path.join(FIGURES_DIR, '{}.pca.png'.format(synthetic_population_code))
    scatterplot(synthetic_population_code, synthetic_principle_components, reference_principle_components, reference_samples, classification_map, class_hierarchy_map, title, xlabel, ylabel, fig_path)

    return synthetic_principle_components, reference_principle_components


def run_umap(synthetic_population_code, synthetic_principle_components, reference_principle_components, reference_samples, classification_map, class_hierarchy_map, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    umap_algorithm = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    umap_algorithm.fit(np.concatenate((reference_principle_components, synthetic_principle_components), axis=0))
    reference_umap_projection = umap_algorithm.transform(reference_principle_components)
    synthetic_umap_projection = umap_algorithm.transform(synthetic_principle_components)
    
    title = 'UMAP'
    xlabel = 'UMAP Dim 1'
    ylabel = 'UMAP Dim 2'
    fig_path = os.path.join(FIGURES_DIR, '{}.umap.png'.format(synthetic_population_code))
    scatterplot(synthetic_population_code, synthetic_umap_projection, reference_umap_projection, reference_samples, classification_map, class_hierarchy_map, title, xlabel, ylabel, fig_path)

    return synthetic_umap_projection, reference_umap_projection


def sfs(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map):
    reference_population_labels = np.array([classification_map.loc[sample]['population'] for sample in reference_samples])
    super_population_groups = class_hierarchy_map.groupby('Super Population Code').groups

    original_reference_genotypes = reference_genotypes[:, reference_population_labels == synthetic_population_code]
    super_population = class_hierarchy_map.loc[synthetic_population_code]['Super Population Code']
    same_super_population_code = super_population_groups[super_population][0] if super_population_groups[super_population][0] != synthetic_population_code else super_population_groups[super_population][1]
    same_super_population_genotypes = reference_genotypes[:, reference_population_labels == same_super_population_code]
    super_populations = list(super_population_groups.keys())
    different_super_population = super_populations[0] if super_populations[0] != super_population else super_populations[1]
    different_super_population_code = super_population_groups[different_super_population][0]
    different_super_population_genotypes = reference_genotypes[:, reference_population_labels == different_super_population_code]

    joint_site_frequency_spectrum(synthetic_genotypes, original_reference_genotypes, 'Synthetic {}'.format(synthetic_population_code), synthetic_population_code)
    joint_site_frequency_spectrum(same_super_population_genotypes, original_reference_genotypes, same_super_population_code, synthetic_population_code)
    joint_site_frequency_spectrum(different_super_population_genotypes, original_reference_genotypes, different_super_population_code, synthetic_population_code)


def joint_site_frequency_spectrum(genotypes1: np.ndarray, genotypes2: np.ndarray, population1: str='population1', population2: str='population2') -> np.ndarray:
    allele_counts1 = genotypes1.reshape(genotypes1.shape[0], -1).sum(1)
    allele_counts2 = genotypes2.reshape(genotypes2.shape[0], -1).sum(1)
    joint_sfs = allel.joint_sfs(allele_counts1, allele_counts2, np.product(genotypes1.shape[1:]), np.product(genotypes2.shape[1:]))
    ax = plot_joint_sfs(joint_sfs, population1, population2)
    plt.savefig(os.path.join(FIGURES_DIR, '{}.{}.joint_sfs.png'.format(population1.replace(' ', '_'), population2.replace(' ', '_'))))
    plt.clf()
    return joint_sfs / joint_sfs.sum()


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
    pos = ax.imshow(s.T, **imshow_kwargs)
    fig.colorbar(pos)

    # tidy
    ax.invert_yaxis()
    ax.set_title('joint site frequency spectrum')
    ax.set_xlabel('derived allele count ({})'.format(population1))
    ax.set_ylabel('derived allele count ({})'.format(population2))

    return ax


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

    # impute missing values with reference allele
    synthetic_genotypes[synthetic_genotypes < 0] = 0
    reference_genotypes[reference_genotypes < 0] = 0

    synthetic_principle_components, reference_principle_components = run_pca(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map)
    run_umap(synthetic_population_code, synthetic_principle_components, reference_principle_components, reference_samples, classification_map, class_hierarchy_map)

    sfs(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map)


if __name__ == '__main__':
    main()
