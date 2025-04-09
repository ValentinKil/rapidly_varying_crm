import numpy as np

# £ File copied from bnplib

#import numpy.random as npr
#from scipy._lib._util import check_random_state

PI=np.pi

def gen_U(w1, w2, w3, gamma, rng):
    V = rng.rand()
    W_p = rng.rand()
    if gamma >= 1:
        if (V < w1/(w1+w2)):
            U = abs(rng.randn()) / np.sqrt(gamma)
        else:
            U = PI * (1 - W_p**2)
    else:
        if (V < w3/(w3 + w2)):
            U = PI * W_p
        else:
            U = PI * (1 - W_p**2)
    return U

def sinc(x):
    return np.sin(x)/x

def ratio_B(x, sigma):
    return sinc(x) / ((sinc(sigma * x))**sigma * (sinc((1-sigma)*x))**(1-sigma))

def zolotarev(u, sigma):
    # Zolotarev function, cf (Devroye, 2009)
    return ((np.sin(sigma*u))**sigma * (np.sin((1-sigma)*u))**(1-sigma) / np.sin(u))**(1/(1-sigma))

def stablernd(alpha, size=(), random_state=None):
    #rng = check_random_state(random_state) # from scipy library
    # cf Devroye, 2009, Equation (2)
    U = random_state.uniform(low=0.0, high = np.pi, size=size)
    E = random_state.exponential(size = size)
    samples = (zolotarev(U, alpha)/E )**((1-alpha)/alpha)
    return samples


def etstablernd(V0, alpha, tau, size=(), random_state=np.random): 
    """
    Generate random samples from an exponentially tilted stable distribution.
    
    Args:
    -----------
    V0 : float
        Scale parameter of the distribution. Must be greater than 0.
    alpha : float
        Stability parameter of the distribution. Must be in the range (0, 1).
    tau : float
        Exponential tilting parameter. Must be greater than or equal to 0.
    size : tuple, optional
        Shape of the output sample array. Default is an empty tuple, which 
        returns a single sample.
    random_state : numpy.random.RandomState, optional
        Random number generator instance. Default is `np.random`.
        
    Returns:
    --------
    samples : ndarray or float
        Random samples from the exponentially tilted stable distribution. If 
        `size` is an empty tuple, a single float is returned. Otherwise, an 
        array of shape `size` is returned.
        
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
        return stablernd(alpha, size=size, random_state=random_state) * V0**(1/alpha)
    
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

    samples = np.zeros(size)
    for i in range(0, len(samples.flat)):
        while True:
            # generate U with density g*/G*
            while True:
                # Generate U with density proportional to g**
                U = gen_U(w1, w2, w3, gamma, random_state)
                while U > PI:
                    U = gen_U(w1, w2, w3, gamma, random_state)
                
                assert U > 0
                assert U <= PI

                W = random_state.rand()
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
            V_p = random_state.rand()
            N_p = random_state.randn()
            E_p = -np.log(random_state.rand())
            if V_p < a1/s:
                X = m - delta*abs(N_p)
            elif V_p < a2/s:
                X = delta * random_state.rand() + m
            else:
                X = m + delta + a3 * E_p

            if X >= 0:
                E = -np.log(random_state.rand())
                cond = (a*(X-m) + np.exp(1/alpha*np.log(lambda_alpha)-b*np.log(m))*((m/X)**b - 1) \
                        - (N_p**2/2) * (X<m) - E_p * (X>m+delta))
                if cond <= E:
                    break
            
        samples.flat[i] = np.exp(1/alpha*np.log(V0) - b*np.log(X)) # more stable than V0^(1/alpha) * X**(-b)

    if size==():
        return samples.flat[0]
    else:
        return samples