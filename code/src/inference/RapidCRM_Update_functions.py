import numpy as np
from scipy.special import gamma, digamma, gammaln

from src.utils.AuxiliaryFunc import safe_exp


# ------------------------------------------------------------------
# Step 1 : Update of the weights w
# ------------------------------------------------------------------

def grad_Uw(N, s, w, w_rem, alpha, tau, beta, c):
    try:
        out = N - s - (beta/c + 2*(np.sum(w) + w_rem)) * w
    except (RuntimeWarning, ValueError, ZeroDivisionError):
        out = -np.inf
    return -out


def grad_Us(N, s, logw, w_rem, alpha, tau, beta, c):
    try:
        out = s * (1-s) * (np.log(c) - logw + digamma(2-s)) + 2 - 4*s
    except (RuntimeWarning, ValueError, ZeroDivisionError):
        out = -np.inf
    return -out


# Ancienne version de la fonction Hamiltonian - kinetic energy has the incorrect term
def Hamiltonian(N, s, u, w, logw, sum_w, w_rem, alpha, tau, beta, c, eta, pw, issimple):
    #
    # u=log(s/(1-s)), imputed for numerical stability
    # logw=log(w), imputed for numerical stability

    # log-joint density given the latent variables, hyperparameters and total mass
    logprob = - (sum_w + w_rem)**2 - beta/c * sum_w + np.sum((N - s) *
                                                             logw + s*np.log(c) - gammaln(2-s) + 4 * np.log(s) - 2 * u)
    # If simple graph, do not take into account self-connections, i.e. w^2 in (- (sum_w + w_rem)**2)
    if issimple:
        logprob = logprob + np.sum(w**2)
    # log density of the pdf of the auxiliary momentum variables

    logauxiliary = -0.5 * np.sum(pw**2)

    potential = - logprob  # potential energy = - log joint density
    kinetic = - logauxiliary  # kinetic energy = - log density of the auxiliary momentum variables
    energy = potential + kinetic  # total energy (hamiltonian)

    return energy, potential, kinetic

# def Hamiltonian_s(N,s,u,w,logw,sum_w,w_rem,alpha,tau,beta,c,eta,ps,issimple):
#     #
#     # u=log(s/(1-s)), imputed for numerical stability
#     # logw=log(w), imputed for numerical stability
#     try:
#         # potential = - (sum_w + w_rem)**2 - beta/c * sum_w + np.sum((N - s) * logw + s*np.log(c) - gammaln(2-s) + 2 * np.log(s) + 2 * np.log(1-s))
#         potential = - (sum_w + w_rem)**2 - beta/c * sum_w + np.sum((N - s) * logw + s*np.log(c) - gammaln(2-s) + 2 * u)
#         if issimple : #If simple graph, do not take into account self-connections (Attention pas sur de comprendre d'où sort ce terme)
#             potential = potential- np.sum(w**2)
#         kinetic = 0.5*np.sum(ps**2)
#         energy=-potential-kinetic
#     except (RuntimeWarning, ValueError, ZeroDivisionError):
#         energy = np.inf
#     return energy


def update_w(s, u, w, logw, w_rem, N, L, epsilon, alpha, tau, beta, c, eta, issimple):
    """
    Update the w of the mGG distribution using Hamiltonian Monte-Carlo algorithm.

    Args:
    ------
        s (numpy.ndarray): The current state.
        u (numpy.ndarray): The current auxiliary variable.
        w (numpy.ndarray): The current weights.
        logw (numpy.ndarray): The current log weights.
        w_rem (float): The remaining weight.
        N (int): The number of particles.
        L (int): The number of leapfrog steps.
        epsilon (float): The leapfrog step size.
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        beta (float): The beta parameter.
        c (float): The c parameter.
        eta (float): The eta parameter.
        issimple (bool): Whether the model is simple or not.

    Returns:
    ------
        tuple: A tuple containing the updated weights, updated log weights, and the acceptance ratio.
    """
    # Proposal
    logwprop = logw.copy()

    # resampling of the momentum  component
    pw = np.random.randn(len(w))

    # Leapfrog scheme
    pwprop = pw - epsilon * grad_Uw(N, s, w, w_rem, alpha, tau, beta, c) / 2
    for lp in range(L-1):
        logwprop += epsilon * pwprop
        wprop = safe_exp(logwprop)
        pwprop -= epsilon * grad_Uw(N, s, wprop, w_rem, alpha, tau, beta, c)

    logwprop += epsilon * pwprop
    wprop = safe_exp(logwprop)
    pwprop -= epsilon * grad_Uw(N, s, wprop, w_rem, alpha, tau, beta, c)/2

    # Acceptance ratio
    sum_wprop = np.sum(wprop)
    sum_w = np.sum(w)
    # added u=log(s/(1-s)) as an argument
    H1, potential1, kinetic1 = Hamiltonian(
        N, s, u, w, logw, sum_w, w_rem, alpha, tau, beta, c, eta, pw, issimple)
    H2, potential2, kinetic2 = Hamiltonian(N, s, u, wprop, logwprop, sum_wprop, w_rem,
                                           # added u=log(s/(1-s)) as an argument
                                           alpha, tau, beta, c, eta, pwprop, issimple)

    logratio = H1-H2
    if np.isnan(logratio):
        print('update_w, logratio is NaN')
        print('H1=', H1)
        print('H2=', H2)
    # Update
    U = np.random.uniform(0, 1)
    if np.log(U) < logratio:
        # print("update")
        w = wprop
        logw = logwprop
        hamil = H2
        potential = potential2
        kinetic = kinetic2
    else:
        hamil = H1
        potential = potential1
        kinetic = kinetic1

    return w, logw, min(1, safe_exp(logratio)), logratio, hamil, potential, kinetic


def update_s(s, u, w, logw, w_rem, N, L, epsilon, alpha, tau, beta, c, eta, issimple):
    """
    Update the s of the mGG distribution using Hamiltonian Monte-Carlo algorithm.

    Args:
    ------
        s (numpy.ndarray): The current state.
        u (numpy.ndarray): The current auxiliary variable.
        w (numpy.ndarray): The current weights.
        logw (numpy.ndarray): The current log weights.
        w_rem (float): The remaining weight.
        N (int): The number of particles.
        L (int): The number of leapfrog steps.
        epsilon (float): The leapfrog step size.
        alpha (float): The alpha parameter.
        tau (float): The tau parameter.
        beta (float): The beta parameter.
        c (float): The c parameter.
        eta (float): The eta parameter.
        issimple (bool): Whether the model is simple or not.

    Returns:
    ------
        tuple: A tuple containing the updated weights, updated log weights, and the acceptance ratio.
    """
    # Proposal
    uprop = u.copy()

    # resampling of the momentum  component
    ps = np.random.randn(len(w))

    # Leapfrog scheme
    psprop = ps - epsilon * grad_Us(N, s, logw, w_rem, alpha, tau, beta, c) / 2
    for lp in range(L-1):
        uprop += epsilon*psprop
        sprop = 1/(1+np.exp(-uprop))
        psprop -= epsilon * grad_Us(N, sprop, logw, w_rem, alpha, tau, beta, c)

    uprop += epsilon*psprop
    sprop = 1/(1+np.exp(-uprop))
    psprop -= epsilon * grad_Us(N, sprop, logw, w_rem, alpha, tau, beta, c)/2

    # Acceptance ratio
    sum_w = np.sum(w)
    # added u=log(s/(1-s)) as an argument
    H1, potential1, kinetic1 = Hamiltonian(
        N, s, u, w, logw, sum_w, w_rem, alpha, tau, beta, c, eta, ps, issimple)
    # added uprop=log(sprop/(1-sprop)) as an argument
    H2, potential2, kinetic2 = Hamiltonian(
        N, sprop, uprop, w, logw, sum_w, w_rem, alpha, tau, beta, c, eta, psprop, issimple)

    logratio = H1-H2
    if np.isnan(logratio):
        print('update_s, logratio is NaN')
        print('H1=', H1)
        print('H2=', H2)
    # Update
    U = np.random.uniform(0, 1)
    if np.log(U) < logratio:
        # print("update")
        s = sprop
        u = uprop
        hamil = H2
        potential = potential2
        kinetic = kinetic2
    else:
        hamil = H1
        potential = potential1
        kinetic = kinetic1

    return s, u, min(1, safe_exp(logratio)), logratio, hamil, potential, kinetic
