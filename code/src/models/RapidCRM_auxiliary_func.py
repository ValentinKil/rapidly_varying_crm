# ----------------------------------------------
# Auxiliary functions for the RapidCRM class
# ----------------------------------------------
import numpy as np
import scipy.integrate as integrate
from scipy.special import gamma
from scipy.optimize import newton
import warnings

from src.utils import LambertW
np_lambertw = LambertW.lambertw_iacono_np


def levy_can(alpha, tau, w):
    """
    Compute the Levy intensity in the canonical case.

    Args:
    ------
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        w (float): The w parameter.

    Returns:
    ------
        float: The computed Levy intensity.
    """
    def foo(s):
        return s/gamma(1-s)*w**(-1-s)
    return 1/(alpha-tau)*integrate.quad(foo, tau, alpha)[0]


def lexp_can_inv(alpha, tau, y, tol=1e-10):
    """
    Compute the inverse of the Laplace exponent in the canonical case.

    Args:
    ------
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        y (float or array-like): The y parameter.
        tol (float, optional): The tolerance level for checking whether y = 1. Defaults to 1e-10.

    Returns:
    ------
        float or array-like: The computed inverse of the Laplace exponent.
    """
    if tau == 0:
        # If tau==0 we have a closed form for the inverse
        y = np.array(y)
        out = np.zeros_like(y)

        indgr1 = (y > 1.0 + tol)
        indless1 = (y < 1.0 - tol)
        indeq1 = np.isclose(y, 1.0, atol=tol)

        out[indgr1] = np.real(-y[indgr1] * np_lambertw(-1 /
                              y[indgr1] * np.exp(-1/y[indgr1]), k=-1))**(1/alpha)
        out[indless1] = np.real(-y[indless1] * np_lambertw(-1 /
                                y[indless1] * np.exp(-1/y[indless1]), k=0))**(1/alpha)
        out[indeq1] = 1.0
        return out

    else:
        # Else we need to use a Newton method
        warnings.warn("Warning: when tau != 0 the inverse is computed with a Newton method", UserWarning)
        def equation_to_solve(t, y_i): return RapidCRMpsi_can(
            t, alpha, tau) - y_i
        t_guess = np.zeros_like(y)
        result = newton(equation_to_solve, t_guess, args=(y,))
        return np.array(result)


def RapidCRMpsi_can(t, alpha, tau):
    """
    Compute the Laplace exponent in the canonical case.

    Args:
    ------
        t (float or array-like): The t parameter.
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.

    Returns:
    ------
        float or array-like: The computed Laplace exponent.
    """
    ones = np.ones_like(t)
    zeros = np.zeros_like(t)

    condition_1 = (t == 1)
    condition_0 = (t == 0)
    condition_else = ~(condition_1 | condition_0)

    result = (
        np.where(condition_1, ones, 0) +
        np.where(condition_0, zeros, 0) +
        np.where(
            condition_else,
            np.divide((t ** alpha - t ** tau), (alpha - tau) * np.log(t),
                      out=np.full_like(t, np.inf, dtype=np.float64), where=(np.log(t) != 0)),
            0
        )
    )
    return result


def RapidCRMpsi(t, alpha, tau, beta, c, eta):
    """
    Compute the Laplce exponent in the general case.

    Args:
    ------
        t (float or array-like): The t parameter.
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        beta (float): The beta parameter.
        c (float): The c parameter.
        eta (float): The eta parameter.

    Returns:
    ---------
        float or array-like: The computed RapidCRMpsi value.
    """
    return eta*(RapidCRMpsi_can(beta+c*t, alpha, tau)-RapidCRMpsi_can(np.array(beta), alpha, tau))


def RapidCRMpsi1(t, alpha, tau, beta,c):
    """
    Compute the Laplace exponent in the general case, when eta is fixed to 1.

    Args:
    ------
        t (float or array-like): The t parameter.
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        beta (float): The beta parameter.
        c (float) : The c parameter 
    Returns:
    --------
        float or array-like: The computed RapidCRMpsi1 value.
    """
    return RapidCRMpsi(t, alpha, tau, beta, c, 1)
