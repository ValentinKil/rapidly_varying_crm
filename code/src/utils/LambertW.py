'''Here we define several version of the lambertw function on the real line.'''


from scipy.special import lambertw
import numpy as np
EXP = np.exp(1)

# %% Lambert W function version scipy.special, so using Halley's method
"""
References
----------
[1] https://en.wikipedia.org/wiki/Lambert_W_function
[2] Corless et al, “On the Lambert W function”, Adv. Comp. Math. 5 (1996) 329-359. https://cs.uwaterloo.ca/research/tr/1993/03/W.pdf
"""

def lambertw_corless(x, k=0):
    """
    Compute the principal branch of the Lambert W function for a given input x.

    Args:
        x (array-like): Input values for which the Lambert W function needs to be computed.
        k (int, optional): Branch index. Default is 0.

    Returns:
        array-like: The computed values of the Lambert W function for the given input x.

    """
    cond1 = (x == (-1/EXP))
    cond2 = ~cond1
    result = np.zeros_like(x)
    result[cond1] = -1
    result[cond2] = lambertw(x[cond2], k=k).real
    return result


# %% Lambert W function version Iacono and Boyd using numpy
"""
References
----------
[1] https://en.wikipedia.org/wiki/Lambert_W_function
[2] Iacono and Boyd, "New approximations to the principal real-valued branch of the Lambert W-function", Adv. Comp. Math. 43 (2017) 1403-1436.
[3] Loczi, "Guaranteed and high precision evaluation of the Lambert W function", Adv. Comp. Math. 433 (2022)
"""


def lambertw_iacono_np(z, k=0, tol=1e-8):
    """
    Compute the Lambert W function using the Iacono method.

    Args:
    ----------
        z (array-like): The input values for which the Lambert W function will be computed.
        k (int, optional): The branch index of the Lambert W function. Default is 0.
        tol (float, optional): The tolerance for convergence. Default is 1e-8.

    Returns:
    ----------
        array-like: The values of the Lambert W function for the given input values.

    Raises:
    ----------
    ValueError: If the argument is not in the domain of the Lambert W function or if the branch index is not valid.
    """

    # Auxiliary functions
    def fun1(x):
        return EXP*x/(1+EXP*x+np.sqrt(1+EXP*x))*np.log(1+np.sqrt(1+EXP*x))

    def fun2(x, w):
        return w/(1+w)*(1+np.log(x/w))

    # Initialization
    result = np.zeros_like(z)

    cond1 = (z > EXP)
    cond2 = (z > 0) & (z < EXP)
    cond3 = (z > -1/EXP) & (z < 0)
    cond4 = (z == -1/EXP)
    cond5 = (z == 0)

    cond6 = ~cond1 & ~cond2 & ~cond3 & ~cond4 & ~cond5

    cond3bis = (z > -1/EXP) & (z <= -1/4)
    cond3ter = (z > -1/4) & (z < 0)

    if np.any(cond6):
        raise ValueError(
            f"The argument is not in the domain of the Lambert W function : {np.where(cond6)}")

    result[cond1] = np.log(z[cond1])-np.log(np.log(z[cond1]))

    result[cond2] = z[cond2]/EXP
    result[cond4] = -1

    if k == 0:
        result[cond3] = fun1(z[cond3])
    elif k == -1:
        result[cond3bis] = -1-np.sqrt(2*(1+EXP*z[cond3bis]))
        result[cond3ter] = np.log(-z[cond3ter])-np.log(-np.log(-z[cond3ter]))
    else:
        raise ValueError(f"The branch index is not valid : {k}")

    # Max number of iterations
    # born1=int(np.log(np.log(tol)/np.log(0.1))/np.log(2))
    b = 1-1/EXP
    N = int(np.log(np.log(5*b*tol)/np.log(b))/np.log(2))

    # Recursive formula
    for i in range(N):
        result[~cond4 & ~cond5] = fun2(
            z[~cond4 & ~cond5], result[~cond4 & ~cond5])

    return result
