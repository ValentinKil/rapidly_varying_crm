# % Import
import numpy as np
from scipy.special import gamma, gammaln
from src.models import CRMclass
from src.sampling.GG_size_biased_rnd import GG_size_biased_sampling
from src.utils.AuxiliaryFunc import generalized_inverse, gammainc_up


class GG(CRMclass.CRM):
    """
    This class represents a Generalized Gamma Process CRM. 
    We take the same notation as in the notes.

    $$
    \rho_{GG}(w;\alpha,\beta,\eta,c)=\frac{\eta/c}{\Gamma(1-\alpha)}(w/c)^{-1-\alpha}e^{-\beta w/c}
    $$

    Args:
    -----
        alpha : float in (0,1) : index of variation.
        beta : positive float : exponential titling exponent.
        eta : positive float : intensity parameter.
        c : positive float : scaling parameter.

    Important Methods:
        lexp(t): Laplace exponent of the GG.
        size_biased_sampling(n): Sample n weights from the GG using Caron2022 equation (15)
    """

    def __init__(self, alpha=1/2, beta=0, c=1, eta=1):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.c = c

    def lexp(self, t):
        return GGpsi(self.c*t, self.eta, self.alpha, self.beta)

    def size_biased_sampling(self, n):
        """
        Sample n weights from the GG using Caron2022 equation (15).

        Args:
        ----
            n : int : number of weights to sample.

        Returns:
            ndarray : array of size n containing the sampled weights.
        """
        return sizeGGrnd(self.alpha, self.beta, self.eta, self.c, n)

    def levy(self, w):
        """
        Compute the Levy intensity of the GG.

        Args:
        ----
            w : float : input value.

        Returns:
        -------
            float : Levy intensity of the GG.
        """
        return self.eta/(self.c*gamma(1-self.alpha))*(w/self.c)**(-1-self.alpha)*np.exp(-self.beta*(w/self.c))

    def AsympNbNodes(self, t):
        """
        Compute the asymptotic number of nodes for rapidCRM.

        Args:
        -----
            t : float : input value.

        Returns:
        -------
            float : asymptotic number of nodes.
        """
        return t**(1+self.alpha)*(2*self.eta*self.c*self.beta**(self.alpha-1))**self.alpha

    def logkappa(self, n, t):
        return gammaln(n-self.alpha)+np.log(self.eta)-gammaln(1-self.alpha)-(n-self.alpha)*np.log(t+self.beta) + self.alpha*np.log(self.c)

    def kappa(self, n, t):
        return np.exp(self.logkappa(n, t))

    def moment(self, n):
        if n-self.alpha <= 0:
            return np.inf
        if self.beta == 0:
            return np.inf
        return self.kappa(n, 0)

    def EspTotal(self):
        """
        Calculates the total expectation of the mGG.

        Returns:
        -------
            float : total expectation.
        """
        if self.beta == 0:
            return np.inf
        return self.moment(1)

    def VarTotal(self):
        """
        Calculates the total variance of the mGG.
        Returns:
        -------
            float : total variance.
        """
        if self.beta == 0:
            return np.inf
        return self.kappa(2, 0)

    def rhobar(self, x):
        if self.beta > 0:
            return self.eta*self.c*(self.beta**self.alpha)*gammainc_up(-self.alpha, self.beta*x/self.c)/gamma(1-self.alpha)
        else:
            return self.eta*self.c*(x/self.c)**(-self.alpha)/gamma(1-self.alpha)/self.alpha

    def rhobarinv(self, y):
        if self.beta > 0:
            mask = (y < 1e4)
            Toreturn = np.zeros_like(y)
            Toreturn[mask] = generalized_inverse(self.rhobar, y[mask])
            Toreturn[~mask] = self.c*(y[~mask]*self.alpha *
                                      gamma(1-self.alpha)/self.eta/self.c)**(-1/self.alpha)
            return Toreturn
        else:
            return self.c*(y*self.alpha*gamma(1-self.alpha)/self.eta/self.c)**(-1/self.alpha)


# ----------------------------------------------
# Auxiliary functions
# ----------------------------------------------


def GGpsi(t, alpha, sigma, tau):
    """
    Compute the Laplace exponant of a GG.

    Args:
    -----
        t : float : input value.
        alpha : float : index of variation.
        sigma : float : exponential titling exponent.
        tau : float : exponential titling parameter.

    Returns:
    -------
        float : value of the function.
    """
    if sigma == 0:  # gamma process
        out = alpha * np.log(1 + t / tau)
    else:
        out = alpha / sigma * ((t + tau) ** sigma - tau ** sigma)
    return out
