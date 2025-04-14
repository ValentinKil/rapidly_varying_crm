import numpy as np
from src.utils import Etstable as etstb
from src.utils import Etstable_accelerated as etstb_acc


def mGGsumrnd(alpha, tau, beta, c, eta, n):
    """
    Compute the total mass of a Rapidly Varying CRM.

    Args:
    ----------
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        beta (float): The beta parameter.
        c (float): The c parameter.
        eta (float): The eta parameter.
        n (int): The number of steps.

    Returns:
    ----------
        float: The total mass of the Rapidly Varying cRM.
    """
    nstep = n
    s = tau+((alpha-tau)/n)*np.arange(0, n)
    if s[0] == 0:
        s = s[1:n]
        nstep = n-1
    # b=(eta/n)**(1/s)
    Toreturn = 0
    for i in range(nstep):
        # X=etstable.etstablernd(1,s[i],beta*b[i],random_state=np.random)
        try:
            X = etstb_acc.etstablernd(eta/n, s[i], beta)
        except (RuntimeWarning, ValueError, ZeroDivisionError):
            print("Acceleration failled")
            print("eta/n=", eta/n)
            print("s[i]=", s[i])
            print("beta=", beta)
            X = etstb.etstablernd(eta/n, s[i], beta)
        # Toreturn +=b[i]*X
        Toreturn += X

    return float(c*Toreturn)
