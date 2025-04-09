import numpy as np
from scipy.optimize import root_scalar
import mpmath as mp

def safe_exp(x, threshold=700):
    """
    Compute the exponential of x, safely handling large values.
    This function returns np.inf for values of x that exceed the threshold.

    Args:
    ----------
        x (float or array-like): Input value(s).
        threshold (float): Threshold beyond which to return np.inf.

    Returns:
    ----------
        float or array-like: Exponential of x or np.inf if x is too large.
    """  
    result = np.full_like(x,np.inf, dtype=float)
    
    # Apply exp only where x<threshold
    mask = x < threshold
    result[mask] = np.exp(x[mask]) 
    return result
 
    

def safe_log(x, threshold = 10**(-320)):
    """
    Compute the natural logarithm of x, safely handling small values.
    This function returns -800 for values of x that are below the threshold.

    Args:
    ----------
        x (numpy.ndarray): Input array.
        threshold (float, optional): Minimum value for log. Defaults to 10**(-320).

    Returns:
    ----------
        numpy.ndarray: Logarithm of `x` or -800 for values below `threshold`.
    """
    result = np.full_like(x, -800, dtype=float)  # Fill with -800 initially
    
    # Apply log only where x >= threshold
    mask = x >= threshold
    result[mask] = np.log(x[mask])    
    return result


def unit_rate_poisson_process(n):
    """
    Generates arrival times for a unit rate Poisson process.

    Args:
    ----------
        n (int): size of the random array to return.

    Returns:
    ----------
        numpy.ndarray: An array of arrival times of size n.
    """

    # Generate exponentially distributed inter-arrival times
    inter_arrival_times = np.random.exponential(scale=1.0, size=n)
    # Cumulative sum to obtain arrival times
    arrival_times = np.cumsum(inter_arrival_times)
    return arrival_times


EXP = np.exp(1)

def generalized_inverse(f, x, method='brentq', tol=1e-16):
    """
    Computes the generalized inverse of a given function `f` for an input value or array `x`.
    
    Args:
    ----------
        f (callable): The function to invert. Must be continuous and strictly monotonic on positive reals.
        x (float or np.ndarray): The value(s) at which to compute the inverse.
        codomain (tuple, optional): A tuple (y_min, y_max) specifying the interval of output values for `f`.
                                    Defaults to None, assuming the function is monotonically increasing.
        method (str, optional): The root-finding method to use ('brentq', 'bisect', or 'secant'). Default is 'brentq'.
        tol (float, optional): The tolerance for the root-finding algorithm. Default is 1e-6.

    Returns:
    ----------
        float or np.ndarray: The generalized inverse of `f` evaluated at `x`.
    """
    def inv_func(y):
        """Finds the inverse of `f` at a single value y."""
        # Define the root-finding problem: find x such that f(x) - y = 0
        def root_eqn(z):
            return f(z) - y
        
        # Find the root using an interval from a small positive number to a large one
        result = root_scalar(root_eqn, bracket=(1e-16, 1e16), method=method, xtol=tol)
        if not result.converged:
            return 0
        return result.root
    
    # Apply inverse function element-wise if `x` is an array
    if isinstance(x, np.ndarray):
        return np.vectorize(inv_func)(x)
    else:
        return inv_func(x)
    


def gammainc_up(a,z):
    '''Computes the upper incomplete gamma function for given parameters.

    The upper incomplete gamma function is defined as:
        Γ(a, z) = ∫[z, ∞] t^(a-1) e^(-t) dt

    Args:
    ----------
        a (float): The shape parameter of the gamma function.
        z (float or array-like): The lower limit of integration. Can be a single value or an array of values.
        
    Returns:
    ----------
        float or numpy.ndarray: The computed upper incomplete gamma function value(s).'''
    try :
        return float(mp.gammainc(a,z))
    except:
        return np.asarray([mp.gammainc(a, zi, regularized=False)
                       for zi in z]).astype(float)