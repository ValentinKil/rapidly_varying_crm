# The functions that update the latent counts n are common to the MCMC algorithm for GGP and for RapidCRM

import numpy as np
import scipy.stats as stats
from scipy.special import gammaln
from scipy.sparse import coo_matrix



def update_n_Gibbs(logw, K, ind1, ind2):
    """
    Update the counts and rates with a Gibbs sampler.

    Args:
    ------
        logw (ndarray): Log weights.
        K (int): Number of clusters.
        ind1 (ndarray): First index array.
        ind2 (ndarray): Second index array.

    Returns:
    ------
        N (ndarray): Total counts for each cluster.
        d (ndarray): Counts for each pair of indices.
        count (ndarray): Count matrix.

    """

    # Calculate the log rate for the Poisson distribution
    lograte_poi = np.log(2) + logw[ind1] + logw[ind2]
    lograte_poi[ind1 == ind2] = 2 * logw[ind1[ind1 == ind2]]

    # Sample from the Poisson distribution
    d = tpoissrnd(np.exp(lograte_poi))

    # Update the count matrix
    count=coo_matrix((d, (ind1, ind2)), shape=(K, K))
    

    # Calculate the total counts for each cluster
    N = np.array(count.sum(axis=0) + count.sum(axis=1).T)[0]
    return N, d, count


def tpoissrnd(lambda_):
    """
    Generate random numbers from a truncated Poisson distribution.

    Args:
    ------
    lambda_ (array-like): The parameter of the Poisson distribution.

    Returns:
    ------
    array-like: Random array generated from the truncated Poisson distribution.

    """
    x = np.ones_like(lambda_)
    ind = lambda_ > 1e-5
    lambda_ind = lambda_[ind]
    x[ind] = stats.poisson.ppf(np.exp(-lambda_ind) + np.random.rand(
        *lambda_ind.shape) * (1 - np.exp(-lambda_ind)), lambda_ind)
    return x

def update_n_MH(logw, d, K, count, ind1, ind2, nbMH):
    """
    Update the values of `d` and `count` using the Metropolis-Hastings algorithm.

    Args:
    ------
    - logw (numpy.ndarray): Array of log weights.
    - d (numpy.ndarray): Array of values.
    - K (int): Number of elements.
    - count (numpy.ndarray): Array of counts.
    - ind1 (numpy.ndarray): Array of indices.
    - ind2 (numpy.ndarray): Array of indices.
    - nbMH (int): Number of Metropolis-Hastings iterations.

    Returns:
    ------
    - N (numpy.ndarray): Updated array of counts.
    - d (numpy.ndarray): Updated array of values.
    - count (numpy.ndarray): Updated array of counts.
    """
    lograte_poi = np.log(2) + logw[ind1] + logw[ind2]
    lograte_poi[ind1 == ind2] = 2 * logw[ind1[ind1 == ind2]]
    for _ in range(nbMH):
        # Metropolis-Hastings update for the latent counts
        ind = (d == 1)
        dprop = d.copy()
        dprop[ind] = 2
        if np.sum(~ind) > 0:
            dprop[~ind] = dprop[~ind] + 2 * np.random.randint(1, 3, size=np.sum(~ind)) - 3

        logqprop = np.zeros_like(ind, dtype=float)
        logqprop[~ind] = np.log(0.5)

        indbis = (dprop == 1)
        logq = np.zeros_like(indbis, dtype=float)
        if np.sum(~indbis) > 0:
            logq[~indbis] = np.log(0.5)

        diff_d = dprop - d
        logaccept_d = (diff_d * lograte_poi 
                       - gammaln(dprop + 1) + gammaln(d + 1)
                       - logqprop + logq)
        
        indaccept = (np.log(np.random.rand(*logaccept_d.shape)) < logaccept_d)
        d[indaccept] = dprop[indaccept]
    # Update the count matrix
    count=coo_matrix((d, (ind1, ind2)), shape=(K, K))
    # Calculate the total counts for each cluster
    N = np.array(count.sum(axis=0) + count.sum(axis=1).T)[0]
    return N, d, count

