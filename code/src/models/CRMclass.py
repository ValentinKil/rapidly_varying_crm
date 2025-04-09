# Import
import numpy as np 
import scipy.integrate as integrate

class CRM:
    """
    This class represents a CRM (Completely Random Measure).

    Args:
    ------
    None

    Methods:
    -------
    - lexp(t): Returns the Laplace exponent of the CRM.
    - levy(w): Returns the Levy intensity of the CRM.
    - step_sampling(): Performs step sampling.
    - size_biased_sampling(): Performs size-biased sampling.
    - CRMsampling(N, Tmax): Performs CRM sampling.
    - EspNbNodes(t, epsilon): Returns the analytical expression of E[N_t], the expected number of nodes at time t.
    - AsympNbNodes(t): Returns the asymptotic number of nodes at time t.
    """

    def lexp(self, t):
        """
        Returns the Laplace exponent of the CRM.

        Args:
        ------
            t: Time parameter.

        """
        pass
    
    def levy(self, w):
        """
        Returns the Levy intensity of the CRM.

        Args:
        ------
        - w: Frequency parameter.

        """
        pass
    
    def step_sampling(self):
        """
        Performs step sampling.
        """
        pass

    def size_biased_sampling(self):
        """
        Performs size-biased sampling.
        """
        pass
    
    def CRMsampling(self, N, Tmax):
        """
        Performs CRM sampling.

        Args:
        ------
        - N: Number of samples.
        - Tmax: Maximum time.

        Returns:
        ------
        The CRM samples as a 2D array.
        """
        W,missing_mass,*_=self.size_biased_sampling(N)
        Theta = np.random.uniform(size=N)*Tmax
        CRM=np.zeros((2,N))
        CRM[0,:]=W
        CRM[1,:]=Theta
        sorted_indices=np.argsort(Theta)
        CRM=CRM[:,sorted_indices]
        return CRM, missing_mass
    
    def EspNbNodes(self, t, epsilon=1e-6):
        """
        Returns the analytical expression of E[N_t], the expected number of nodes at time t.

        Args:
        ------
        - t: Time parameter.
        - epsilon: Small value for numerical integration (default: 1e-6).

        Returns:
        ------
        - float : the expected number of nodes at time t.
        """
        def func(w, t):
            return (1 - np.exp(-w**2 - t * self.lexp(2*w))) * self.levy(w)
        t_array = np.asarray(t)
        result_vectorized = np.vectorize(lambda ti: ti * integrate.quad(func, epsilon, np.inf, args=(ti,), limit=100)[0])(t_array)

        return result_vectorized
    
    def AsympNbNodes(self, t):
        """
        Returns the asymptotic number of nodes at time t.

        Args:
        ------
        - t: Time parameter.
        
        """
        pass