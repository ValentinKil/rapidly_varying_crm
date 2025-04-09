import numpy as np
from src.utils import AuxiliaryFunc


def GG_size_biased_sampling(alpha, beta, eta, c, n):
    '''
    This function generates size-biased samples from the Generalized Gamma Process (GG).
    Args:
    -----
       alpha: shape parameter of the GG
       beta: scale parameter of the GG
       eta: intensity parameter of the GG
       c: constant factor for scaling the samples
       n: number of samples to generate

    Returns:
    -------
       samples: array of size-biased samples from the GG
        '''
    epsilon = AuxiliaryFunc.unit_rate_poisson_process(n)
    samples = c*np.random.gamma(shape=1-alpha, scale=1 /
                                ((alpha/eta*epsilon+beta**alpha)**(1/alpha)))
    missing_mass = GG_missing_mass_size_sample(alpha, eta, c, n)
    return samples, missing_mass


def GG_missing_mass_size_sample(alpha, eta, c, n):
    missing_mass = c*eta*(eta/alpha/n)**(1/alpha-1)
    return missing_mass
