import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict
from tqdm import tqdm


import src.inference.mGG_Update_functions as rapid_up_func
from src.inference import mGG_Update_param
from src.inference import Update_counts


def Rapidgraphmcmc(G, modelparam, mcmcparam, typegraph, nmass,
                   true_value=None, initial_value=None, verbose=True):
    """
    Perform Markov Chain Monte Carlo (MCMC) inference for a Rapidly Varying graph model.

    Args:
    ------
        G: sparse logical adjacency matrix
        modelparam: structure of model parameters with the following fields:
            - alpha : scalar
            - beta : scalar
            - beta: if scalar, the value of beta. If vector of length 2, parameters of the gamma prior over beta
            - c: if scalar, the value of c. If vector of length 2, parameters of the gamma prior over c
            - eta: if scalar, the value of eta. If vector of length 2, parameters of the gamma prior over eta
        mcmcparam: structure of MCMC parameters with the following fields:
            - niter: number of MCMC iterations
            - nburn: number of burn-in iterations
            - thin: thinning of the MCMC output
            - leapfrog.L: number of leapfrog steps
            - leapfrog.epsilon: leapfrog stepsize
            - latent.MH_nb: number of MH iterations for latent (if 0: Gibbs update)
            - hyper.MH_nb: number of MH iterations for hyperparameters
            - hyper.rw_std: standard deviation of the random walk
            - store_w: logical. If true, returns MCMC draws of w
        typegraph: type of graph ('undirected' or 'simple')
        nmass : number of iteration in the estimation of the remaining mass
        true_value : defaultdict containing the true value of some parameters, if a true value is provided for a parameter, this parameter will not be estimated and it will be fixed to this true value
        initiale_value : defaultdict containing the true value of some parameters
        verbose: logical. If true (default), print information

    Returns:
    ------
        samples: structure with the MCMC samples for the variables
            - w
            - w_rem
            - beta
            - c
            - eta
        tats: structure with summary stats about the MCMC algorithm
            - rate: acceptance rate of the HMC step at each iteration
            - rate2: acceptance rate of the MH for the hyperparameters at each iteration

    See also:
    - graphmcmc
    - graphmodel
    """

    """# Check if the graph is symmetric, sparse, and logical
    if not (G==G.T).all():
        raise ValueError("Adjacency matrix G must be a symmetric.")
    if not isinstance(G, coo_matrix) :
        raise ValueError("Adjacency matrix G must be a sparse.")
    if not np.issubdtype(G.dtype, np.bool_):
        raise ValueError("Adjacency matrix G must be a logical.") """

    if true_value is None:
        true_value = defaultdict(lambda: None)
    if initial_value is None:
        initial_value = defaultdict(lambda: None)

    if typegraph == 'simple':
        issimple = True
    else:
        issimple = False

    # Extract model parameters
    alpha = modelparam['alpha']
    tau = modelparam['tau']

    # Extract MCMC parameters
    niter = mcmcparam['niter']
    nburn = mcmcparam['nburn']
    thin = mcmcparam['thin']
    L = mcmcparam['leapfrog']['L']
    epsilon = mcmcparam['leapfrog']['epsilon']  # / G.shape[0]**(1/4)

    if initial_value['epsilon_w0']:
        epsilon_w = initial_value['epsilon_w0']
    else:
        epsilon_w = epsilon
    if initial_value['epsilon_s0']:
        epsilon_s = initial_value['epsilon_s0']
    else:
        epsilon_s = epsilon

    # HMC with Dual averaging for s
    # epsilon_s=0.1
    # epsilon_bar_s=1
    # mu_s=np.log(10*epsilon_s)
    # H_s=0
    # gamma_s=0.05
    # t_0s=10
    # kappa_s=0.75

    latent_MH_nb = mcmcparam['latent']['MH_nb']
    hyper_MH_nb = mcmcparam['hyper']['MH_nb']
    hyper_rw_std = mcmcparam['hyper']['rw_std']
    store_w = mcmcparam['store_w']

    # Initialize hyperparameters

    '''
    If len(modelparam['beta'])=2 then the prior on beta is gamma distribution of parameters modelparam['beta'].
    In that case beta is initialize as N(10,1) and we will estimate beta.
    If len(modelparam['beta'])=1 then beta is set at modelparam['beta'] and we will not estimate beta.
    Same for c and eta
    '''

    if not isinstance(modelparam['beta'], float):
        if initial_value['beta0']:
            beta = initial_value['beta0']
        else:
            beta = 1.0  # 10 * np.random.rand()
        estimate_beta = True
    else:
        beta = modelparam['beta']
        estimate_beta = False

    if not isinstance(modelparam['c'], float):
        if initial_value['c0']:
            c = initial_value['c0']
        else:
            c = 1.0  # 10 * np.random.rand()
        estimate_c = True
    else:
        c = modelparam['c']
        estimate_c = False

    if not isinstance(modelparam['eta'], float):
        if initial_value['eta0']:
            eta = initial_value['eta0']
        else:
            eta = 10  # 100 * np.random.rand()
        estimate_eta = True
    else:
        eta = modelparam['eta']
        estimate_eta = False

    # Make sur that the adjacency matrix has the right shape
    K = G.shape[0]

    # Extract COO attributes
    rows, cols, data = G.row, G.col, G.data

    # Create the combined indices for G + G.T symmetry
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.concatenate([data, data])
    G_sym = coo_matrix((sym_data, (sym_rows, sym_cols)), shape=(K, K))
    G_sym.sum_duplicates()

    sym_rows, sym_cols, sym_data = G_sym.row, G_sym.col, G_sym.data

    # Compute strict or general upper triangular mask
    if issimple:
        upper_triangular_mask = sym_rows < sym_cols
    else:
        upper_triangular_mask = sym_rows <= sym_cols

    upper_row = sym_rows[upper_triangular_mask]
    upper_col = sym_cols[upper_triangular_mask]
    upper_data = sym_data[upper_triangular_mask]

    G_upper = coo_matrix((upper_data, (upper_row, upper_col)), shape=(K, K))

    # Apply the final mask to get the indices
    ind1, ind2 = G_upper.nonzero()

    # Initialize counts
    if true_value['N_true'] is None:
        estimate_count = True
        if initial_value['count_0'] is None:
            n = np.random.randint(1, 11, len(ind1))  # pourquoii 11 ?
            count = coo_matrix((n, (ind1, ind2)), shape=(K, K))
            N1 = np.array(count.sum(axis=1)).flatten()
            N2 = np.array(count.sum(axis=0)).flatten()
            N = N1 + N2
        else:
            N, n, count = initial_value['tupleCount_0']
    else:
        N = true_value['N_true']
        estimate_count = False

    # Initialize weight w
    if true_value['w_true'] is None:
        estimate_w = True
        if initial_value['w0'] is None:
            w = np.random.rand(K)
        else:
            w = initial_value['w0']
        logw = np.log(w)
    else:
        estimate_w = False
        w = true_value['w_true']
        logw = np.log(w)

    # Initialize latent s
    if true_value['s_true'] is None:
        estimate_s = True
        if initial_value['s0'] is None:
            s = np.random.rand(K)
        else:
            s = initial_value['s0']
        u = np.log(s / (1.0 - s))
    else:
        estimate_s = False
        s = true_value['s_true']
        u = np.log(s / (1.0 - s))

    if true_value['w_rem_true'] is None:
        estimate_w_rem = True
        if initial_value['w_rem0'] is None:
            w_rem = .1  # np.random.gamma(1, 1)
        else:
            w_rem = initial_value['w_rem0']
    else:
        estimate_w_rem = False
        w_rem = true_value['w_rem_true']

    # Initialize the rate:
    if not initial_value['rates0']:
        rates0 = 0
    else:
        rates0 = initial_value['rates0']

    if not initial_value['ratew0']:
        ratew0 = 0
    else:
        ratew0 = initial_value['ratew0']

    if not initial_value['size0']:
        size0 = 0
    else:
        size0 = initial_value['size0']

    # Choice of update for the latent (Gibbs/MH)
    if latent_MH_nb == 0:
        # Gibbs update
        # if verbose:
        #     print("Gibbs")

        def update_n(logw, d, K, count, ind1, ind2):
            return Update_counts.update_n_Gibbs(logw, K, ind1, ind2)
    else:
        # Metropolis-Hastings update
        # if verbose:
        #     print("MH")

        def update_n(logw, d, K, count, ind1, ind2):
            return Update_counts.update_n_MH(
                logw, d, K, count, ind1, ind2, latent_MH_nb)

    # To store MCMC samples
    n_samples = (niter - nburn) // thin
    if store_w:
        w_st = np.zeros((n_samples, K))
        s_st = np.zeros((n_samples, K))
    else:
        w_st = []
        s_st = []

    w_rem_st = np.zeros(n_samples)
    beta_st = np.zeros(n_samples)
    c_st = np.zeros(n_samples)
    eta_st = np.zeros(n_samples)

    rate_w = np.zeros(niter)
    rate_s = np.zeros(niter)
    rate2 = np.zeros(niter)
    logratio_w = np.zeros(niter)
    logratio_s = np.zeros(niter)
    epsilon_s_st = np.zeros(niter)
    epsilon_w_st = np.zeros(niter)
    hamiltonianhmc_st = np.zeros(niter)
    potentialhmc_st = np.zeros(niter)
    kinetichmc_st = np.zeros(niter)

    if verbose:
        print("-------------------------------")
        print("Start MCMC for mGG graphs")
        print(f"Nb of nodes:", K, "- Nb of edges:", G_upper.sum() / 2)
        print("Estimate beta: ", estimate_beta, "- Estimate eta:",
              estimate_eta, "- Estimate c:", estimate_c)
        print(
            "Estimate count:",
            estimate_count,
            "- Estimate weights:",
            estimate_w,
            " - Estimate s:",
            estimate_s,
            "- Estimate w_rem:",
            estimate_w_rem)
        print(f"Number of iterations:", niter)
        print("Intiale value: beta0=", beta, "- c0=",
              c, "- eta0=", eta, '- w_rem_0=', w_rem)
        print("-------------------------------")

    display_info = niter // 1000

    # Initialize tqdm progress bar
    progress_bar = tqdm(range(niter))
    for i in progress_bar:
        if verbose and (i % display_info == 0) and (i > 0):
            progress_bar.set_description((
                f"hamil={hamiltonianhmc_st[i - 1]:.2f}, poten={potentialhmc_st[i - 1]:.2f}, kine={kinetichmc_st[i - 1]:.2f}, "
                f"beta={beta:.2f}, c={c:.2f}, eta={eta:.2f}, w_rem={w_rem:.2f}, "
                f"eps_w={epsilon_w:.3f}, eps_s={epsilon_s:.3f}, avrate_w={np.average(rate_w[0:i]):.2f}, avrate_s={np.average(rate_s[0:i]):.2f}, "
                f"minlogw={np.min(logw):.1e}, maxlogw={np.max(logw):.2f}, "
                # f"mins={np.min(s):.1e}, maxs={np.max(s):.2f}, "
                f"minsigs={np.min(u):.1e}, maxsigs={np.max(u):.1e}"
            ))

        # Update weights using Hamiltonian Monte Carlo
        if estimate_w:
            #
            epsilon_w_st[i] = epsilon_w

            w, logw, rate_w[i], logratio_w[i], hamiltonianhmc_st[i], potentialhmc_st[i], kinetichmc_st[i] = rapid_up_func.update_w(
                s, u, w, logw, w_rem, N, L, epsilon_w, alpha, tau, beta, c, eta, issimple)

            # check_w(w,i)

            if size0 + i < mcmcparam['leapfrog']['nadapt']:
                current_rate_w = (
                    size0 * ratew0 + np.average(rate_w[0:i + 1]) * (i + 1)) / (size0 + i + 1)
                epsilon_w = np.exp(np.log(epsilon_w) +
                                   0.005 * (current_rate_w - 0.65))

        if estimate_s:
            epsilon_s_st[i] = epsilon_s

            s, u, rate_s[i], logratio_s[i], _, _, _ = rapid_up_func.update_s(
                s, u, w, logw, w_rem, N, L, epsilon_s, alpha, tau, beta, c, eta, issimple)
            # Check s for None, NaN, or inf values

            # check_s(s, i)

            if size0 + i < mcmcparam['leapfrog']['nadapt']:
                current_rate_s = (
                    size0 * rates0 + np.average(rate_s[0:i + 1]) * (i + 1)) / (size0 + i + 1)
                epsilon_s = np.exp(np.log(epsilon_s) +
                                   0.005 * (current_rate_s - 0.65))

        # Update w_rem and hyperparameters using Metropolis-Hastings
        # Half of the time alpha is update using Random Walk Metropolis Hasting

        if i % 2 == 0:
            rw_eta = True
        else:
            rw_eta = False

        w_rem, alpha, tau, beta, c, eta, rate2[i] = mGG_Update_param.update_hyper(
            w, s, w_rem, alpha, tau, beta, c, eta, hyper_MH_nb, hyper_rw_std, estimate_beta, modelparam["beta"], estimate_c, modelparam["c"], estimate_eta, modelparam["eta"], rw_eta, estimate_w_rem, nmass)

        # Update of the count
        if estimate_count and i % 30 == 0:
            N, n, count = update_n(logw, n, K, count, ind1, ind2)

        if np.isnan(eta):
            raise ValueError("eta is NaN")

        # if i == 10:
            # print("-------------------------------")
            # print("Start MCMC for mGG graphs")
            # print(f"Nb of nodes:", K, "- Nb of edges:", G_upper.sum()/2)
            # print("Estimate beta: ",estimate_beta,"- Estimate eta:",estimate_eta,"- Estimate c:", estimate_c)
            # print("Estimate count:", estimate_count,"- Estimate weights:",estimate_w," - Estimate s:",estimate_s, "- Estimate w_rem:",estimate_w_rem)
            # print(f"Number of iterations:", niter)
            # print("-------------------------------")
        if (i >= nburn) and (i - nburn) % thin == 0:
            ind = (i - nburn) // thin
            if store_w:
                w_st[ind] = w
                s_st[ind] = s
            w_rem_st[ind] = w_rem
            beta_st[ind] = beta
            c_st[ind] = c
            eta_st[ind] = eta

    samples = {
        'w': w_st,
        'w_rem': w_rem_st,
        'beta': beta_st,
        'c': c_st,
        'eta': eta_st,
        's': s_st,
    }

    stats = {
        'rate_w': rate_w,
        'rate_s': rate_s,
        'rate2': rate2,
        'logratio_w': logratio_w,
        'logratio_s': logratio_s,
        'epsilon_s': epsilon_s_st,
        'epsilon_w': epsilon_w_st,
        'hamiltonianHMC': hamiltonianhmc_st,
        'potentialHMC': potentialhmc_st,
        'kineticHMC': kinetichmc_st
    }

    last = {
        'w': w,
        'w_rem': w_rem,
        'beta': beta,
        'c': c,
        'eta': eta,
        's': s,
        'epsilon_s': epsilon_s,
        'epsilon_w': epsilon_w,
        'tupleCount': (N, n, count),
        'rate_w': rate_w[-1],
        'rate_s': rate_s[-1],
    }

    print("-------------------------------")
    print("End MCMC for mGG graphs")
    print("-------------------------------")

    return samples, stats, last
