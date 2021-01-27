import itertools
import os
import sys

import allel
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, squareform
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


def plot_pairwise_ld(m1, m2, colorbar=True, ax=None, imshow_kwargs=None):
    """Plot a matrix of genotype linkage disequilibrium values between
    all pairs of variants.

    Parameters
    ----------
    m1 : array_like
        Array of linkage disequilibrium values in condensed form.
    m2 : array_like
        Array of linkage disequilibrium values in condensed form.
    colorbar : bool, optional
        If True, add a colorbar to the current figure.
    ax : axes, optional
        The axes on which to draw. If not provided, a new figure will be
        created.
    imshow_kwargs : dict-like, optional
        Additional keyword arguments passed through to
        :func:`matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : axes
        The axes on which the plot was drawn.

    """

    import matplotlib.pyplot as plt

    # check inputs
    m1_square = allel.util.ensure_square(m1)
    m2_square = allel.util.ensure_square(m2)

    # blank out upper triangle
    m1_square = np.triu(m1_square)
    m2_square = np.tril(m2_square)
    m_square = m1_square + m2_square

    # set up axes
    if ax is None:
        # make a square figure with enough pixels to represent each variant
        x = m_square.shape[0] / plt.rcParams['figure.dpi']
        x = max(x, plt.rcParams['figure.figsize'][0])
        fig, ax = plt.subplots(figsize=(x, x))
        fig.tight_layout(pad=0)

    # setup imshow arguments
    if imshow_kwargs is None:
        imshow_kwargs = dict()
    imshow_kwargs.setdefault('interpolation', 'none')
    imshow_kwargs.setdefault('cmap', 'Greys')
    imshow_kwargs.setdefault('vmin', 0)
    imshow_kwargs.setdefault('vmax', 1)

    # plot as image
    im = ax.imshow(m_square, **imshow_kwargs)

    # tidy up
    ax.set_xticks([])
    ax.set_yticks([])
    for s in 'bottom', 'right':
        ax.spines[s].set_visible(False)
    if colorbar:
        plt.gcf().colorbar(im, shrink=.5, pad=0)

    return ax


def binned_ld(genotypes, positions, window_size, num_bins=20):
    bins = dict((i, []) for i in range(num_bins))
    exponent_start = 8
    base = np.exp(np.log(window_size) / (exponent_start + num_bins))

    def bin_index(pos1, pos2):
        dist = np.abs(pos2 - pos1)
        return int(max(np.floor(np.log(dist) / np.log(base) - exponent_start), 0))

    for window_start in range(positions[0], positions[-1], window_size):
        window_indices = np.logical_and(positions >= window_start, positions < window_start + window_size)
        window_positions = positions[window_indices]
        window_gn = genotypes[window_indices]

        if len(window_positions) == 0:
            continue
        r = allel.rogers_huff_r(window_gn)
        r_squared_matrix = squareform(r ** 2)

        for i, j in itertools.combinations(range(len(window_positions)), 2):
            r_squared = r_squared_matrix[i, j]
            if np.isnan(r_squared):
                continue
            index = bin_index(window_positions[i], window_positions[j])
            bins[index].append(r_squared)

    sizes = [base ** i for i in range(exponent_start + 1, exponent_start + num_bins + 1)]
    binned_r_squared = [np.mean(bins[i]) for i in range(num_bins)]
    return sizes, binned_r_squared


def remove_fixed_sites(genotypes, positions):
    minor_allele_count = genotypes.sum(1)
    fixed_site_removal_indices = np.logical_and(minor_allele_count != 0, minor_allele_count != genotypes.shape[1] * 2)
    genotypes = genotypes[fixed_site_removal_indices]
    positions = positions[fixed_site_removal_indices]
    return genotypes, positions


def ld(synthetic_population_code, synthetic_genotypes, reference_genotypes, synthetic_positions, reference_positions, reference_samples, classification_map, window_size=2e5):
    window_size = int(window_size)
    reference_population_labels = np.array([classification_map.loc[sample]['population'] for sample in reference_samples])
    original_reference_genotypes = reference_genotypes[:, reference_population_labels == synthetic_population_code]

    synthetic_genotypes, synthetic_positions = remove_fixed_sites(allel.GenotypeArray(np.copy(synthetic_genotypes)).to_n_alt(), np.copy(synthetic_positions))
    reference_genotypes, reference_positions = remove_fixed_sites(allel.GenotypeArray(np.copy(original_reference_genotypes)).to_n_alt(), np.copy(reference_positions))

    # # plot binned ld
    plt.title('Binned Linkage Disequilibrium')
    sizes, binned_r_squared = binned_ld(synthetic_genotypes, synthetic_positions, window_size)
    plt.plot(sizes, binned_r_squared, label='Synthetic {}'.format(synthetic_population_code))
    sizes, binned_r_squared = binned_ld(reference_genotypes, reference_positions, window_size)
    plt.plot(sizes, binned_r_squared, label='{}'.format(synthetic_population_code))
    plt.xlabel('Distance (bp)')
    plt.ylabel('LD (r squared)')
    plt.xscale('log')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, '{}.binned_ld.png'.format(synthetic_population_code)))
    plt.close(plt.gcf())

    # plot pairwise ld
    np.random.seed(SEED)
    window_start = np.random.randint(synthetic_positions[0], synthetic_positions[-1] - window_size)
    synthetic_window_indices = np.logical_and(np.logical_and(synthetic_positions >= window_start, synthetic_positions < window_start + window_size), np.isin(synthetic_positions, reference_positions))
    reference_window_indices = np.logical_and(np.logical_and(reference_positions >= window_start, reference_positions < window_start + window_size), np.isin(reference_positions, synthetic_positions))
    synthetic_window_gn = synthetic_genotypes[synthetic_window_indices]
    reference_window_gn = reference_genotypes[reference_window_indices]
    synthetic_r = allel.rogers_huff_r(synthetic_window_gn)
    reference_r = allel.rogers_huff_r(reference_window_gn)
    synthetic_r_squared_matrix = squareform(synthetic_r ** 2)
    reference_r_squared_matrix = squareform(reference_r ** 2)
    ax = plot_pairwise_ld(synthetic_r_squared_matrix, reference_r_squared_matrix, colorbar=True, imshow_kwargs={'cmap': 'cividis'})
    plt.title('SNP Correlation in {}kb Window'.format(window_size // 1000))
    plt.savefig(os.path.join(FIGURES_DIR, '{}.pairwise_ld.png'.format(synthetic_population_code)))
    plt.close(plt.gcf())


def population_statistics(synthetic_population_code, synthetic_genotypes, reference_genotypes, synthetic_positions, reference_positions, reference_samples, classification_map, window_size=2e5):
    window_size = int(window_size)
    reference_population_labels = np.array([classification_map.loc[sample]['population'] for sample in reference_samples])
    original_reference_genotypes = reference_genotypes[:, reference_population_labels == synthetic_population_code]

    synthetic_allele_counts = allel.GenotypeArray(synthetic_genotypes).count_alleles()
    reference_allele_counts = allel.GenotypeArray(original_reference_genotypes).count_alleles()

    synthetic_pi, _, _, _ = allel.windowed_diversity(synthetic_positions, synthetic_allele_counts, size=window_size)
    reference_pi, _, _, _ = allel.windowed_diversity(reference_positions, reference_allele_counts, size=window_size)

    plt.title('Nucleotide Diversity Sliding Window Analysis')
    plt.plot(np.arange(1, len(synthetic_pi) + 1), synthetic_pi, label='Synthetic {}'.format(synthetic_population_code))
    plt.plot(np.arange(1, len(reference_pi) + 1), reference_pi, label='{}'.format(synthetic_population_code))
    plt.xlabel('Windows ({}kb)'.format(window_size // 1000))
    plt.ylabel('Nucleotide Diversity (π)')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, '{}.pi.png'.format(synthetic_population_code)))
    plt.close(plt.gcf())

    synthetic_D, _, _ = allel.windowed_tajima_d(synthetic_positions, synthetic_allele_counts, size=window_size)
    reference_D, _, _ = allel.windowed_tajima_d(reference_positions, reference_allele_counts, size=window_size)

    plt.title('Tajima\'s D Sliding Window Analysis')
    plt.plot(np.arange(1, len(synthetic_D) + 1), synthetic_D, label='Synthetic {}'.format(synthetic_population_code))
    plt.plot(np.arange(1, len(reference_D) + 1), reference_D, label='{}'.format(synthetic_population_code))
    plt.xlabel('Windows ({}kb)'.format(window_size // 1000))
    plt.ylabel('Tajima\'s D')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, '{}.tajima_d.png'.format(synthetic_population_code)))
    plt.close(plt.gcf())


def nearest_neighbor_adversarial_accuracy(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map):
    reference_population_labels = np.array([classification_map.loc[sample]['population'] for sample in reference_samples])
    original_reference_genotypes = reference_genotypes[:, reference_population_labels == synthetic_population_code]

    assert(synthetic_genotypes.shape[0] == original_reference_genotypes.shape[0])

    synthetic_data = synthetic_genotypes.reshape(synthetic_genotypes.shape[0], -1).T
    reference_data = original_reference_genotypes.reshape(original_reference_genotypes.shape[0], -1).T

    D_tt = cdist(reference_data, reference_data)
    np.fill_diagonal(D_tt, np.inf)
    D_ss = cdist(synthetic_data, synthetic_data)
    np.fill_diagonal(D_ss, np.inf)
    D_ts = cdist(reference_data, synthetic_data)
    D_st = D_ts.T

    d_tt = D_tt.min(1)
    d_ss = D_ss.min(1)
    d_ts = D_ts.min(1)
    d_st = D_st.min(1)

    print(d_tt)
    print(d_ss)
    print(d_ts)
    print(d_st)

    AA_true = np.mean(d_ts > d_tt)
    AA_syn = np.mean(d_st > d_ss)
    AA_ts = .5 * (AA_true + AA_syn)

    print('AA_true: {:.3f}'.format(AA_true))
    print('AA_syn: {:.3f}'.format(AA_syn))
    print('AA_ts: {:.3f}'.format(AA_ts))


def ld_pruning():
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

    # population_statistics(synthetic_population_code, synthetic_genotypes, reference_genotypes, synthetic_positions, reference_positions, reference_samples, classification_map)

    # impute missing values with reference allele
    synthetic_genotypes[synthetic_genotypes < 0] = 0
    reference_genotypes[reference_genotypes < 0] = 0

    # synthetic_principle_components, reference_principle_components = run_pca(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map)
    # run_umap(synthetic_population_code, synthetic_principle_components, reference_principle_components, reference_samples, classification_map, class_hierarchy_map)

    # sfs(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map, class_hierarchy_map)
    # ld(synthetic_population_code, synthetic_genotypes, reference_genotypes, synthetic_positions, reference_positions, reference_samples, classification_map, window_size=5e4)
    nearest_neighbor_adversarial_accuracy(synthetic_population_code, synthetic_genotypes, reference_genotypes, reference_samples, classification_map)


if __name__ == '__main__':
    main()
