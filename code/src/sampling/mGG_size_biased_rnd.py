import numpy as np
import warnings

from src.models.mGG_auxiliary_func import mGGpsi_can, lexp_can_inv
from src.utils import AuxiliaryFunc, LambertW


EXP = np.exp(1)

# Choice of the method to compute the LambertW function
np_lambertw = LambertW.lambertw_iacono_np


def step_sampling(alpha, tau, beta, c, eta, epsi, u):
    """
    Perfom the sampling of one weight in the size_biased sampling scheme based on the provided values of epsi and u.

    Args:
    --------
            alpha (float): The alpha parameter.
            tau (float): The tau parameter.
            beta (float): The beta parameter.
            c (float): The c parameter.
            eta (float): The eta parameter.
            epsi (float): Value of a unit rate Poisson process.
            u (float): Value of a uniform variable.

    Returns:
    --------
            float: The result of the step sampling process.
            float: The latent value S.
    """

    y = epsi / eta + mGGpsi_can(np.array(beta), alpha, tau)
    T = lexp_can_inv(alpha, tau, y) - beta
    S = F_inv(alpha, tau, beta, u, T)
    Wprime = np.random.gamma(shape=1-S, scale=1/(T+beta))
    return c * Wprime, S


def mGG_size_biased_sampling(alpha, tau, beta, c, eta, n):
    """
    Perform size-biased sampling using the mGG algorithm.

    Args:
    --------
            alpha (float): The alpha parameter.
            tau (float): The tau parameter.
            beta (float): The beta parameter.
            c (float): The c parameter.
            eta (float): The eta parameter.
            n (int): The number of weights to generate.

    Returns:
    --------
            numpy.ndarray: Weight of the CRM.
            np.float : Approximation of the mass miss by the sampling procedure
            numpy.ndarray: The lattent values S.

    """
    # Sample weights
    epsilon = AuxiliaryFunc.unit_rate_poisson_process(n)
    uni = np.random.uniform(size=n)
    W, S = step_sampling(alpha, tau, beta, c, eta, epsilon, uni)

    # Compute missing mass
    if tau == 0 and alpha == 1:
        missing_mass = mGG_missing_mass_size_sample(eta, c, n)
    else:
        warnings.warn(
            "Missing mass is not implemented for cases where tau != 0 or alpha != 1. Setting missing_mass to 0.")
        missing_mass = 0

    return W, missing_mass, S

# --------------------------------------------
# Auxiliarly function
# --------------------------------------------


def c_aux(alpha, tau, y, z):
    Toreturn = (z**tau-z**alpha+(alpha*z**alpha-tau*z**tau)
                * np.log(z))*y-z**tau + tau*z**tau*np.log(z)
    return Toreturn


def F_inv(alpha, tau, beta, y, t, tol=1e-10):
    # Inverse cdf of the probability measure p(s|t)

    z = t+beta

    cond1 = np.isclose(z, 1.0, atol=tol)
    cond2 = (z > 1.0 + tol)
    cond3 = (z < 1.0-tol)

    result = np.zeros_like(z)

    result[cond1] = np.sqrt((alpha**2 - tau**2) * y[cond1] + tau**2)
    result[cond2] = ((np_lambertw(c_aux(alpha, tau, y[cond2],
                     z[cond2]) / EXP, k=0) + 1) / np.log(z[cond2]))
    result[cond3] = ((np_lambertw(c_aux(alpha, tau, y[cond3],
                     z[cond3]) / EXP, k=-1) + 1) / np.log(z[cond3]))
    return result


def mGG_missing_mass_size_sample(eta, c, n):
    # Return the missing mass in size sampling
    # from asymptotics in section 3.5.2 - more accurate as log(n) not so large
    missing_mass = c * eta * (np.log(n/eta * np.log(n)) - 1) / \
        np.log(n/eta * np.log(n))**2
    return missing_mass
