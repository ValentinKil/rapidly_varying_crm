import numpy as np
from tqdm import tqdm
from src.display.plotdegreedistribut import plot_degree
from src.models.mGGclass import mGG
from src.sampling.CaronFoxGraph_rnd import CRMtoGraph


def plot_posterior_degree_statistics(objmcmc, nburn, niter, thin, p, q, ndraws=2000, bins=16, N=10**5):
    """
    Generate and analyze posterior predictive for our graph model. This function samples from 
    the posterior distribution of our graph model and generates random graphs to compute 
    various statistics of interest, such as the number of nodes, number of edges, proportion 
    of nodes with degree one, and mean degree.

    Args:
    -----------
    objmcmc : object
        An object containing MCMC samples and settings. It must have attributes `settings` 
        (a dictionary with the number of chains) and `samples` (a list of sampled parameters).
        Additionally, it must have a `prior` attribute with a `type` field indicating the 
        type of graph model (e.g., 'Rapid').
    nburn : int
        The number of burn-in iterations to discard from the MCMC samples.
    niter : int
        The total number of MCMC iterations.
    thin : int
        The thinning interval for the MCMC samples.
    p : float
        A parameter related to the graph model.
    q : float
        A parameter related to the graph model.
    ndraws : int, optional (default=2000)
        The number of posterior predictive samples to generate.
    bins : int, optional (default=16)
        The number of bins to use when computing degree distributions.
    N : int, optional (default=10**5)
        The size parameter for generating random graphs.

    Returns:
    --------
    Nbnodes : numpy.ndarray
        An array containing the number of nodes for each sampled graph.
    NbEdges : numpy.ndarray
        An array containing the number of edges for each sampled graph.
    PropDegreeOnes : numpy.ndarray
        An array containing the proportion of nodes with degree one for each sampled graph.
    Meandegree : numpy.ndarray
        An array containing the mean degree for each sampled graph.
    """

    nchains = objmcmc.settings["nchains"]
    n_samples = (niter - nburn) // thin

    # Initialize frequency array to store degree distributions
    Nbnodes = np.zeros(ndraws)
    NbEdges = np.zeros(ndraws)
    PropDegreeOnes = np.zeros(ndraws)
    Meandegree = np.zeros(ndraws)

    index = np.random.randint(0, nchains, size=ndraws)
    INDEX = nburn//thin+np.random.randint(0, n_samples, size=ndraws)

    if objmcmc.prior.type == 'Rapid':
        print("Rapid")
        beta_samples = np.array([objmcmc.samples[i]["beta"][idx]
                                for i, idx in zip(index, INDEX)])
        c_samples = np.array([objmcmc.samples[i]["c"][idx]
                             for i, idx in zip(index, INDEX)])
        eta_samples = np.array(
            [objmcmc.samples[i]["eta"][idx]*q*(1-p)/p for i, idx in zip(index, INDEX)])
        Arg = tuple(zip(np.ones(ndraws), np.zeros(ndraws),
                    beta_samples, c_samples, eta_samples))
        typeCRM = mGG
    # elif objmcmc.prior.type == 'GG':
    #     print("GG")
    #     alpha_samples = np.array([objmcmc.samples[i]["alpha"][idx]*q*(1-p)/p for i, idx in zip(index, INDEX)])
    #     tau_samples = np.array([objmcmc.samples[i]["tau"][idx] for i, idx in zip(index, INDEX)])
    #     sigma_samples = np.array([objmcmc.samples[i]["sigma"][idx] for i, idx in zip(index, INDEX)])
    #     Arg= tuple(zip(sigma_samples,tau_samples,np.ones(ndraws),alpha_samples))
    #     typeCRM = GG
    else:
        raise ValueError(f"Unsuported type of graph {objmcmc.prior.typegraph}")

    # Generate degree distributions from posterior predictive
    for i in tqdm(range(ndraws)):
        if i % 200 == 0:
            print(f'Sample {i}/{ndraws} from the posterior predictive')

        PPP = typeCRM(*Arg[i])

        # Generate random graph based on the sampled parameters
        w_CRM, missing_mass, *_ = PPP.size_biased_sampling(N)
        G_samp = CRMtoGraph(w_CRM, missing_mass, store=False)

        # Compute the statistics of interest
        Nbnodes[i] = G_samp.shape[0]
        NbEdges[i] = np.sum(G_samp.data)/2
        centerbins, freq, deg = plot_degree(G_samp, bins=bins, display=False)
        PropDegreeOnes[i] = freq[0]
        Meandegree[i] = np.average(deg)

    return Nbnodes, NbEdges, PropDegreeOnes, Meandegree
