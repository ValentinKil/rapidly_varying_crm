'''This file computes exponentially tilted stable (etstable) random variables efficiently. 
It leverages Numba's Just-In-Time (JIT) compilation to accelerate the computations.'''

from numba import njit
import numpy as np

PI=np.pi

@njit
def gen_U(w1: float, w2: float, w3: float, gamma: float) -> float:
    V = np.random.rand()
    W_p = np.random.rand()
    if gamma >= 1:
        if (V < w1/(w1+w2)):
            U = abs(np.random.randn()) / np.sqrt(gamma)
        else:
            U = PI * (1 - W_p**2)
    else:
        if (V < w3/(w3 + w2)):
            U = PI * W_p
        else:
            U = PI * (1 - W_p**2)
    return U

@njit
def sinc(x: float) -> float:
    return np.sin(x)/x

@njit
def ratio_B(x : float, sigma: float ) -> float:
    return sinc(x) / ((sinc(sigma * x))**sigma * (sinc((1-sigma)*x))**(1-sigma))

@njit
def zolotarev(u: np.ndarray, sigma : float) -> float:
    # Zolotarev function, cf (Devroye, 2009)
    return ((np.sin(sigma*u))**sigma * (np.sin((1-sigma)*u))**(1-sigma) / np.sin(u))**(1/(1-sigma))

@njit()
def stablernd(alpha: float) -> float:
    # cf Devroye, 2009, Equation (2)
    U = np.random.uniform(0.0,PI)
    E = np.random.exponential()
    samples = (zolotarev(U, alpha) / E)**((1 - alpha) / alpha)
    return samples

@njit()
def etstablernd(V0: float, alpha: float, tau: float) -> float:
    """
    Generate random samples from an exponentially tilted stable distribution. 
    It leverages Numba's Just-In-Time (JIT) compilation to accelerate the computations.
    
    Args:
    -----------
    V0 : float
        Scale parameter of the distribution. Must be greater than 0.
    alpha : float
        Stability parameter of the distribution. Must be in the range (0, 1).
    tau : float
        Exponential tilting parameter. Must be greater than or equal to 0.
        
    Returns:
    --------
    samples : float
        Random samples from the exponentially tilted stable distribution. 
        
    References:
    -----------
    [1] Devroye, L. "Random variate generation for exponentially and 
      polynomially tilted stable distributions". ACM Transactions on Modeling 
      and Computer Simulation (2009).
    [2] Hofert, M.  "Sampling exponentially tilted stable distributions". ACM Transactions on Modeling and Computer Simulation (2011), 
      55(1), 154-157.
    """
    # random_state should be an instance of numpy.random.RandomState
    
    # check params
    if alpha <= 0 or alpha >= 1:
         raise ValueError('alpha must be in ]0,1[')
    if tau < 0:
         raise ValueError('tau must be >= 0')
    if V0 <= 0:
        raise ValueError('V0 must be > 0')
    
    # if V0.size!=1:
    #     raise ValueError('Broadcast parameters not implemented yet') 

    if tau==0:
        return stablernd(alpha) * V0**(1/alpha)
    
    lambda_alpha = tau**alpha * V0

    # Now we sample from an exponentially tilted distribution of parameters
    # alpha, lambda, as in (Devroye, 2009)
    gamma = lambda_alpha * alpha * (1-alpha)

    xi = 1/PI *((2+np.sqrt(PI/2)) * np.sqrt(2*gamma) + 1) # Correction in Hofert
    psi = 1/PI * np.exp(-gamma * PI**2/8) * (2 + np.sqrt(PI/2)) * np.sqrt(gamma * PI)
    w1 = xi * np.sqrt(PI/(2*gamma))
    w2 = 2 * psi * np.sqrt(PI)
    w3 = xi * PI
    b = (1-alpha)/alpha

    while True:
        # generate U with density g*/G*
        while True:
            # Generate U with density proportional to g**
            U = gen_U(w1, w2, w3, gamma)
            while U > PI:
                U = gen_U(w1, w2, w3, gamma)
            
            assert U > 0
            assert U <= PI

            W = np.random.rand()
            zeta = np.sqrt(ratio_B(U, alpha))
            try : 
                z = 1/(1 - (1 + alpha*zeta/np.sqrt(gamma))**(-1/alpha))
            except:
                print("lambda_alpha=",lambda_alpha)
                print("tau=",tau)
                print("V0=",V0)
                print("alpha=",alpha)
                print("zeta=",zeta)
                print("gamma=",gamma)
            rho = 1
            rho = PI * np.exp(min(-lambda_alpha * (1-zeta**(-2)), 1e+2)) \
                    * (xi * np.exp(-gamma*U**2/2) * (gamma>=1) \
                    + psi/np.sqrt(PI-U) \
                    + xi * (gamma<1)) \
                    /((1 + np.sqrt(PI/2))*np.sqrt(gamma)/zeta + z)

            if W*rho <= 1:
                break
            

        # Generate X with density proportional to g(x, U)
        a = zolotarev(U, alpha)
        m = (b/a)**alpha * lambda_alpha
        delta = np.sqrt(m*alpha/a)
        a1 = delta * np.sqrt(PI/2)
        a2 = a1 + delta # correction in Hofert
        a3 = z/a
        s = a1 + delta + a3 # correction in Hofert
        V_p = np.random.rand()
        N_p = np.random.randn()
        E_p = -np.log(np.random.rand())
        if V_p < a1/s:
            X = m - delta*abs(N_p)
        elif V_p < a2/s:
            X = delta * np.random.rand() + m
        else:
            X = m + delta + a3 * E_p

        if X >= 0:
            E = -np.log(np.random.rand())
            cond = (a*(X-m) + np.exp(1/alpha*np.log(lambda_alpha)-b*np.log(m))*((m/X)**b - 1) \
                    - (N_p**2/2) * (X<m) - E_p * (X>m+delta))
            if cond <= E:
                break
        
    return np.exp(1/alpha*np.log(V0) - b*np.log(X)) # more stable than V0^(1/alpha) * X**(-b)