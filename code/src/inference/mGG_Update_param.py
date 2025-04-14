import numpy as np
import scipy.stats as stats

from src.sampling import mGG_totalmass_rnd
from src.models.mGG_auxiliary_func import mGGpsi1
from src.utils.AuxiliaryFunc import safe_exp

# ------------------------------------------------------------------
# Step 2 : update of alpha,tau,beta, eta, and w_*
# ------------------------------------------------------------------


def update_hyper(w, s, w_rem, alpha, tau, beta, c, eta, nbMH, rw_std, estimate_beta, hyper_beta, estimate_c, hyper_c, estimate_eta, hyper_eta, rw_eta, estimate_w_rem, nmass):
    """
    Update the hyperparameters of the GG distribution using Metropolis-Hastings algorithm.

    Args:
    ------
        w (numpy.ndarray): Array of weights.
        logw (numpy.ndarray): Array of logarithm of weights.
        w_rem (float): Remaining weight.
        alpha (float): Hyperparameter alpha.
        tau (float): Hyperparameter tau.
        beta (float): Hyperparameter beta.
        c (float): Hyperparameter c.
        eta (float): Hyperparameter eta.
        nbMH (int): Number of Metropolis-Hastings iterations.
        rw_std (numpy.ndarray): Array of random walk standard deviations.
        estimate_alpha (bool): Flag indicating whether to estimate alpha.
        estimate_tau (bool): Flag indicating whether to estimate tau.
        estimate_beta (bool): Flag indicating whether to estimate beta.
        estimate_c (bool): Flag indicating whether to estimate c.
        estimate_eta (bool): Flag indicating whether to estimate eta.
        hyper_beta (numpy.ndarray): Array of hyperparameters for beta.
        hyper_c (numpy.ndarray): Array of hyperparameters for c.
        hyper_eta (numpy.ndarray): Array of hyperparameters for eta.
        rw_eta (bool): Flag indicating whether to use random walk for eta.
        nmass (int): Number of mass points for w_rem.
        estimate_w_rem (bool): Flag indicating whether to estimate w_rem.

    Returns:
    ------
            tuple: Tuple containing the updated values of w_rem, alpha, tau, beta, c, eta, and the acceptance rate.
    """

    K = len(w)
    for _ in range(nbMH):
        sum_w = np.sum(w)

        # Estimate of beta
        if estimate_beta:
            betaprop = beta * np.exp(rw_std[0] * np.random.randn())
        else:
            betaprop = beta

        # Estimate of c
        if estimate_c:
            cprop = c * np.exp(rw_std[1] * np.random.randn())  # Random walk
            # cprop = (c+rw_std[1]**2/2*(beta*sum_w/c**2+np.sum(s)/c))*np.exp(rw_std[1] * np.random.randn())  #Langevin

        else:
            cprop = c

        # Estimate of eta
        if estimate_eta:
            if not rw_eta:
                try:
                    etaprop = stats.gamma.rvs(
                        K, scale=1 / mGGpsi1(2 * sum_w + 2 * w_rem, alpha, tau, betaprop, cprop))
                except:
                    print(mGGpsi1(2 * sum_w + 2 * w_rem,
                          alpha, tau, betaprop, cprop))
            else:
                etaprop = eta * np.exp(rw_std[2] * np.random.randn())
        else:
            etaprop = eta

        # Estimate of w_rem
        if estimate_w_rem:
            wprop_rem = mGG_totalmass_rnd.mGGsumrnd(
                alpha, tau, betaprop + 2 * cprop * sum_w + 2 * cprop * w_rem, cprop, etaprop, nmass)
        else:
            wprop_rem = w_rem

        # Acceptance ratio
        # sumall_wprop = sum_wprop + wprop_rem

        term1 = - ((sum_w + wprop_rem)**2 - (sum_w + w_rem)
                   ** 2)  # - sumall_wprop**2 + sumall_w**2
        term2 = - (betaprop/cprop - beta/c + 2 * (w_rem - wprop_rem)) * sum_w
        if not rw_eta:
            term3 = K * (np.log(mGGpsi1((2 * sum_w + 2 * wprop_rem), alpha, tau, beta, c)) -
                         np.log(mGGpsi1((2 * sum_w + 2 * w_rem), alpha, tau, betaprop, cprop)))
        else:
            term3 = K * (np.log(etaprop) - np.log(eta)) - etaprop * mGGpsi1((2 * sum_w + 2 * w_rem), alpha,
                                                                            tau, betaprop, cprop) + eta * mGGpsi1((2 * sum_w + 2 * wprop_rem), alpha, tau, beta, c)

        if estimate_c:
            term4 = np.sum(s) * np.log(cprop/c) + \
                hyper_c[0]*(np.log(cprop)-np.log(c))-hyper_c[1]*(cprop-c)
            # term4 += -(np.log(cprop)-np.log(c+rw_std[1]**2/2*(beta*sum_w/c**2+np.sum(s)/c)))**2/(2*rw_std[1]**2)+(np.log(c)-np.log(cprop+rw_std[1]**2/2*(beta*sum_w/cprop**2+np.sum(s)/cprop)))**2/(2*rw_std[1]**2) - np.log(cprop/c) #Langevin

        else:
            term4 = 0
        if estimate_beta:
            term5 = hyper_beta[0]*(np.log(betaprop)-np.log(beta)) - \
                hyper_beta[1]*(betaprop-beta)  # Gamma prior on beta
        else:
            term5 = 0

        logaccept = term1 + term2 + term3 + term4+term5

        if np.isnan(logaccept):
            logaccept = -np.inf

        if np.log(np.random.rand()) < logaccept:
            w_rem = wprop_rem
            beta = betaprop
            c = cprop
            eta = etaprop

    rate2 = min(1, safe_exp(logaccept))
    return w_rem, alpha, tau, beta, c, eta, rate2
