import torch


sse_loss = torch.nn.MSELoss(reduction='sum')


def vae_loss(x: torch.FloatTensor, x_hat: torch.FloatTensor, mu: torch.FloatTensor, logvar: torch.FloatTensor):
    sse = sse_loss(x_hat, x)
    kld = kld_loss(mu, logvar)
    return sse + kld, sse, kld

def kld_loss(mu: torch.FloatTensor, logvar: torch.FloatTensor):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

def cov_loss(cov: torch.FloatTensor, x: torch.FloatTensor):
    pass

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

def corr_coef(m: torch.FloatTensor, rowvar: bool=True):
    covariance = cov(m, rowvar)
    correlation_coefficients = covariance / covariance.diag().unsqueeze(1).matmul(covariance.diag().unsqueeze(0)).sqrt()
    return correlation_coefficients

def linkage_disequilibrium_correlation(genotype: torch.FloatTensor) -> torch.FloatTensor:
    allele_prob = genotype.mean(0)
    joint_allele_prob = genotype.unsqueeze(-1).matmul(genotype.unsqueeze(1)).mean(0)
    allele_prob_product = allele_prob.unsqueeze(-1).matmul(allele_prob.unsqueeze(0))
    allele_prob_product[range(allele_prob_product.shape[0]), range(allele_prob_product.shape[1])] = allele_prob
    disequilibrium = joint_allele_prob - allele_prob_product
    intermediary_denominator_term = allele_prob * (1 - allele_prob)
    correlation_denominator = intermediary_denominator_term.unsqueeze(-1).matmul(intermediary_denominator_term.unsqueeze(0)).sqrt()
    correlation = disequilibrium / correlation_denominator
    correlation[correlation.isnan()] = 0
    return correlation

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
