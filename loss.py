from typing import Tuple

import torch


bcel_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
sigmoid = torch.nn.Sigmoid()


def vae_loss(x: torch.FloatTensor, logits: torch.FloatTensor, mu: torch.FloatTensor, logvar: torch.FloatTensor, window_size: int) -> torch.FloatTensor:
    reconstruction = reconstruction_loss(x, logits, window_size)
    kld = kld_loss(mu, logvar)
    return reconstruction + kld, reconstruction, kld

def reconstruction_loss(x: torch.FloatTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
    # return x to 0 and 1 encoding
    x = ((x + 1) * (.5)).round()
    # sum across variants and mean across batch
    likelihood = bcel_loss(logits, x).sum(1).mean()
    # site_frequency = site_frequency_loss(x, logits)
    # ld = linkage_disequilibrium_loss(x, logits, window_size)
    # print(site_frequency.detach())
    # print(ld.detach())
    return likelihood
    # return likelihood * (1 + site_frequency + ld)

def kld_loss(mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.FloatTensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()

def site_frequency_loss(x: torch.FloatTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
    allele_freq = x.mean(0)
    x_mean, x_var, x_skew, x_kurt = profile_distribution(allele_freq)
    x_distribution = torch.cat([x_mean.unsqueeze(-1), x_var.unsqueeze(-1), x_skew.sign() * x_skew.pow(1. / 3.).unsqueeze(-1), x_kurt.pow(1. / 4.).unsqueeze(-1)], -1)

    # turn logits into probability values (mean of probs will be mean of approximated data)
    probs = sigmoid(logits)
    x_hat_mean = probs.mean(0).mean()

    # turn logits into more extreme probability values to simulate sampling
    x_hat = sigmoid(logits * 10)
    allele_freq_hat = x_hat.mean(0)

    _, x_hat_var, x_hat_skew, x_hat_kurt = profile_distribution(allele_freq_hat)
    x_hat_distribution = torch.cat([x_hat_mean.unsqueeze(-1), x_hat_var.unsqueeze(-1), x_hat_skew.sign() * x_hat_skew.abs().pow(1. / 3.).unsqueeze(-1), x_hat_kurt.pow(1. / 4.).unsqueeze(-1)], -1)

    # print(x_distribution.detach())
    # print(x_hat_distribution.detach())
    # print(x_distribution.dist(x_hat_distribution).detach())

    return x_distribution.dist(x_hat_distribution)

def linkage_disequilibrium_loss(x: torch.FloatTensor, logits: torch.FloatTensor, window_size: int) -> torch.FloatTensor:
    windowed_r_squared = windowed_squared_corr_coef(x.T, window_size)
    windowed_r_squared_hat = windowed_squared_corr_coef(logits.sigmoid().T, window_size)

    ld_losses = []
    for r_squared, r_squared_hat in zip(windowed_r_squared, windowed_r_squared_hat):
        nan_indices = r_squared.isnan().logical_or(r_squared_hat.isnan())
        non_nan_r_squared = r_squared[~nan_indices]
        if len(non_nan_r_squared) == 0:
            continue
        non_nan_r_squared_hat = r_squared_hat[~nan_indices]
        dist_of_mean = non_nan_r_squared.mean().dist(non_nan_r_squared_hat.mean())
        dist_of_var = non_nan_r_squared.var().dist(non_nan_r_squared_hat.var())
        ld_losses.append((dist_of_mean + dist_of_var).unsqueeze(0))

    if len(ld_losses) == 0:
        return 0

    mean_ld_loss = torch.cat(ld_losses).mean()

    return mean_ld_loss

def cov(m: torch.FloatTensor, rowvar: bool=True, inplace: bool=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def squared_corr_coef(m: torch.FloatTensor, rowvar: bool=True) -> torch.FloatTensor:
    covariance = cov(m, rowvar)
    squared_correlation_coefficients = covariance.pow(2) / covariance.diag().unsqueeze(1).matmul(covariance.diag().unsqueeze(0))
    return squared_correlation_coefficients

def windowed_squared_corr_coef(m: torch.FloatTensor, window_size: int, rowvar: bool=True) -> Tuple[torch.FloatTensor, ...]:
    return tuple([squared_corr_coef(window, rowvar).unsqueeze(0) for window in m.split(window_size)])

def var(x: torch.FloatTensor, mean: torch.FloatTensor=None) -> torch.FloatTensor:
    if mean is None:
        mean = x.mean()
    return (x.pow(2) - mean.pow(2)).sum() * (1 / (len(x) - 1))

def skew(x: torch.FloatTensor, mean: torch.FloatTensor=None) -> torch.FloatTensor:
    if mean is None:
        mean = x.mean()
    return ((x - mean) / x.std()).pow(3).mean()

def kurtosis(x: torch.FloatTensor, mean: torch.FloatTensor=None) -> torch.FloatTensor:
    if mean is None:
        mean = x.mean()
    return ((x - mean) / x.std()).pow(4).mean()

def profile_distribution(x: torch.FloatTensor, mean: torch.FloatTensor=None) -> torch.FloatTensor:
    if mean is None:
        mean = x.mean()
    return torch.cat([mean.unsqueeze(-1), var(x, mean).unsqueeze(-1), skew(x, mean).unsqueeze(-1), kurtosis(x, mean).unsqueeze(-1)], -1)

def hellinger(a: torch.FloatTensor, b: torch.FloatTensor) -> torch.FloatTensor:
    return (1 / torch.tensor(2, dtype=torch.float).sqrt()) * a.sqrt().dist(b.sqrt())

def linkage_disequilibrium_correlation(genotype: torch.FloatTensor) -> torch.FloatTensor:
    allele_prob = genotype.mean(0)
    joint_allele_prob = genotype.unsqueeze(-1).matmul(genotype.unsqueeze(1)).mean(0)
    allele_prob_product = allele_prob.unsqueeze(-1).matmul(allele_prob.unsqueeze(0))
    allele_prob_product[range(allele_prob_product.shape[0]), range(allele_prob_product.shape[1])] = allele_prob
    disequilibrium = joint_allele_prob - allele_prob_product
    intermediary_denominator_term = allele_prob * (1 - allele_prob)
    correlation_denominator = intermediary_denominator_term.unsqueeze(-1).matmul(intermediary_denominator_term.unsqueeze(0))
    r_squared = disequilibrium.pow(2) / correlation_denominator
    return r_squared

# def linkage_disequilibrium_correlation(genotype: torch.FloatTensor) -> torch.FloatTensor:
#     allele_prob = genotype.mean(0)
#     joint_allele_prob = genotype.unsqueeze(-1).matmul(genotype.unsqueeze(1)).mean(0)
#     allele_prob_product = allele_prob.unsqueeze(-1).matmul(allele_prob.unsqueeze(0))
#     allele_prob_product[range(allele_prob_product.shape[0]), range(allele_prob_product.shape[1])] = allele_prob
#     disequilibrium = joint_allele_prob - allele_prob_product
#     intermediary_denominator_term = allele_prob * (1 - allele_prob)
#     correlation_denominator = intermediary_denominator_term.unsqueeze(-1).matmul(intermediary_denominator_term.unsqueeze(0)).sqrt()
#     correlation = disequilibrium / correlation_denominator
#     correlation[correlation.isnan()] = 0
#     return correlation
