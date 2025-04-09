import numpy as np
import scipy.stats as stats
from src.utils import Etstable


def GGsumrnd(alpha, sigma, tau):
    '''
    Compute the total mass of a Generalized Gamma process.

    Args:
    -----
        alpha (float): Shape parameter of the Generalized Gamma process.
        sigma (float): Scale parameter of the Generalized Gamma process.
        tau (float): Shape parameter of the Generalized Gamma process.

    Returns:
    ------
        float: Total mass of the Generalized Gamma process.

    '''
    if sigma < -1e-8:
        # Compound Poisson case
        # S is distributed from a Poisson mixture of gamma variables
        K = stats.poisson.rvs(-alpha / sigma / tau ** (-sigma))
        try:
            S = stats.gamma.rvs(-sigma * K, scale=1 / tau)
        except:
            S = 0
    elif sigma < 1e-8:
        # Gamma process case
        # S is gamma distributed
        S = stats.gamma.rvs(alpha, scale=1 / tau)
    elif sigma == 0.5 and tau == 0:
        # Inverse Gaussian process case
        # S is distributed from an inverse Gaussian distribution
        lambda_ = 2 * alpha ** 2
        mu = alpha / np.sqrt(tau)
        S = stats.invgauss.rvs(mu, scale=lambda_)
    else:
        # General case
        # S is distributed from an exponentially tilted stable distribution
        S = Etstable.etstablernd(alpha/sigma, sigma, tau,
                                 size=1, random_state=np.random)[0]

    return S
