import numpy as np
from scipy.special import gamma
import warnings

from src.models import CRMclass
from src.models.RapidCRM_auxiliary_func import RapidCRMpsi_can, levy_can, RapidCRMpsi
from src.sampling.RapidCRM_size_biased_rnd import RapidCRM_size_biased_sampling
from src.utils import LambertW
np_lambertw = LambertW.lambertw_iacono_np



class RapidCRM(CRMclass.CRM):
    """
    This class represents a Rapidly Varying CRM as described in the note.

    Args:
    -------
        alpha : float in (0,1] : index of regular variation.
        tau : float in [0,alpha] : parameter representing something.
        beta : positive float : exponential titling exponent.
        c : positive float : scaling parameter.
        eta : positive float : rate parameter.

    Important Methods:
     -------------
        lexp(t): Calculates the Laplace exponent of the CRM.
        size_biased_sampling(n): Samples n weights from the CRM using the method described in the note.
    """

    def __init__(self, alpha=1, tau=0, beta=0, c=1, eta=1):
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.c = c
        self.eta = eta

    def lexp_can(self, t):
        return RapidCRMpsi_can(t, self.alpha, self.tau)

    def lexp(self, t):
        return RapidCRMpsi(t, self.alpha, self.tau, self.beta, self.c, self.eta)

    def levy_can(self, w):
        return levy_can(self.alpha, self.tau, w)

    def levy(self, w):
        return self.eta/self.c*self.levy_can(w/self.c)*np.exp(-self.beta/self.c*w)

    def levy_vectorized(self, w_values):
        """
        Calculates the Levy intensity of the CRM for an array of input values.

        Args:
        -------
            w_values : array-like : input values.
            
        Returns:
        -------
            array-like : Levy intensity of the CRM for each input value.
        """
        levy_can_vectorized = np.vectorize(lambda w: self.levy_can(w / self.c))
        return self.eta / self.c * levy_can_vectorized(w_values/self.c) * np.exp(-self.beta / self.c * w_values)

    def size_biased_sampling(self, n):
        """
        Samples n weights from the CRM using the size-biased sampling method.

        Args:
        -------
            n : int : number of weights to sample.

        Returns:
        -------
            array-like : sampled weights.
        """
        return RapidCRM_size_biased_sampling(self.alpha, self.tau, self.beta, self.c, self.eta, n)

    def const(self):
        """
        Calculates the constant that appears in the asymptotics of the Rapidly Varying CRM.

        Returns:
        -------
            float : constant value.
        """
        if self.beta == 1:
            return (self.eta*self.c)**2
        else:
            return 2*(self.eta*self.c)**2*(1/self.beta-1+np.log(self.beta))/(np.log(self.beta)**2)

    def AsympNbNodes(self, t):
        """
        Calculates the asymptotic number of nodes for the RapidCRM.

        Args:
        -------
            t : float : input value.

        Returns:
        -------
            float : asymptotic number of nodes.
        """
        return t**2*self.const()/np.log(t)

    def EspTotal(self):
        """
        Calculates the total expectation of the RapidCRM.

        Returns:
        -------
            float : total expectation.
        """
        if self.beta == 1:
            return self.eta*self.c*(self.alpha+self.tau)/2
        else:
            return self.eta*self.c*(self.beta**(-1)/(self.alpha-self.tau)*((self.beta**self.tau-self.beta**self.alpha+(self.alpha*self.beta**self.alpha-self.tau*self.beta**self.tau)*np.log(self.beta))/(np.log(self.beta)**2)))

    def VarTotal(self):
        """
        Calculates the total variance of the RapidCRM.

        Returns:
        -------
            float : total variance.
        """
        if self.alpha != 1 or self.tau != 0:
            warnings.warn("Attention, implemented only for alpha=1 and tau=0", UserWarning)
        if self.beta == 1:
            return self.eta*self.c**2/6
        else:
            return self.eta*self.c**2*(((self.beta+1)*np.log(self.beta+2*(1-self.beta)))/(self.beta**2*np.log(self.beta)**3))
        
    def rhobarinvasympt(self,t):
        if self.alpha==1:
            return self.eta*self.c/(1-self.tau)/t/np.log(t)**2
        else:
            C=self.eta*self.c**self.alpha/(self.alpha-self.tau)/gamma(1-self.alpha)
            return (-self.alpha*C/t/np_lambertw(-self.alpha*C/t,k=-1))**(1/self.alpha)
            
        
        
        
