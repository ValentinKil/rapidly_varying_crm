import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from src.display.plotdegreedistribut import plot_degree
from src.models.RapidCRMclass import RapidCRM
from src.sampling.CaronFoxGraph_rnd import CRMtoGraph


def plot_posterior_predictive_degree(objmcmc,nburn,niter,thin, G=None,p=None, ndraws=2000, bins=16, N=10**5,plot_folder=None,fontsize=16,legend=True,dpi=300, figsize=(6, 8),xlim=(0, 2*10**3),ylim=(8*10**(-8),1)):
    """
    Plots the posterior predictive degree distribution for a given MCMC object.
    This function generates degree distributions from posterior predictive samples
    of a specified graph model and visualizes the 95% credible intervals along with the 
    empirical degree distribution if a graph `G` is provided.
    
    Args:
    -----------
    objmcmc : object
        The MCMC object containing posterior samples and settings.
    nburn : int
        Number of burn-in iterations to discard.
    niter : int
        Total number of iterations in the MCMC chain.
    thin : int
        Thinning interval for the MCMC chain.
    G : networkx.Graph, optional
        The empirical graph for which the degree distribution is computed and compared.
        Default is None.
    p : float, optional
        A parameter used to adjust the eta_samples in the 'Rapid' prior case. Default is None.
    ndraws : int, optional
        Number of posterior predictive samples to draw. Default is 2000.
    bins : int, optional
        Number of bins for the degree distribution histogram. Default is 16.
    N : int, optional
        Number of nodes for the generated random graph. Default is 10**5.
    plot_folder : str, optional
        Path to the folder where the plot will be saved. If None, the plot is not saved.
        Default is None.
    fontsize : int, optional
        Font size for the plot labels and title. Default is 16.
    legend : bool, optional
        Whether to display the legend on the plot. Default is True.
 
    Return:
    --------
    - A plot showing the posterior predictive degree distribution with 95% credible intervals.
    - If `G` is provided, the empirical degree distribution is overlaid on the plot.
    - The plot is saved to `plot_folder` if specified.
    """
    
    nchains=objmcmc.settings["nchains"]
    n_samples = (niter - nburn) // thin
    
    # Initialize frequency array to store degree distributions
    FREQ = np.zeros((ndraws, bins))
 
    index=np.random.randint(0,nchains,size=ndraws)
    INDEX=nburn//thin+np.random.randint(0,n_samples,size=ndraws)
    
    if objmcmc.prior.type == 'Rapid':
        print("Rapid")
        beta_samples = np.array([objmcmc.samples[i]["beta"][idx] for i, idx in zip(index, INDEX)])
        c_samples = np.array([objmcmc.samples[i]["c"][idx] for i, idx in zip(index, INDEX)])
        if p is None:
            eta_samples = np.array([objmcmc.samples[i]["eta"][idx] for i, idx in zip(index, INDEX)])
        else:
            eta_samples = np.array([objmcmc.samples[i]["eta"][idx] for i, idx in zip(index, INDEX)])*(1-p)/p
        Arg = tuple(zip(np.ones(ndraws),np.zeros(ndraws),beta_samples,c_samples,eta_samples))
        typeCRM = RapidCRM
    # elif objmcmc.prior.type == 'GGP':
    #     print("GGP")
    #     alpha_samples = np.array([objmcmc.samples[i]["alpha"][idx] for i, idx in zip(index, INDEX)])
    #     tau_samples = np.array([objmcmc.samples[i]["tau"][idx] for i, idx in zip(index, INDEX)])
    #     sigma_samples = np.array([objmcmc.samples[i]["sigma"][idx] for i, idx in zip(index, INDEX)])
    #     Arg= tuple(zip(sigma_samples,tau_samples,np.ones(ndraws),alpha_samples))
    #     typeCRM = GGP
    else:
        raise ValueError(f"Unknown type of graph {objmcmc.prior.typegraph}")
    

    # Generate degree distributions from posterior predictive
    for i in tqdm(range(ndraws)):
        if i % 200 == 0:
            print(f'Sample {i}/{ndraws} from the posterior predictive')

        PPP=typeCRM(*Arg[i])

        # Generate random graph based on the sampled parameters
        w_CRM,missing_mass, *_ = PPP.size_biased_sampling(N)
        G_samp= CRMtoGraph(w_CRM,missing_mass,store=False)


        # Compute the degree distribution
        centerbins, freq, deg = plot_degree(G_samp,bins=bins,display=False)
        FREQ[i, :] = freq


    # Compute 2.5% and 97.5% quantiles (95% credible interval)
    quantile_freq = np.quantile(FREQ, [0.025, 0.975], axis=0)

    # Compute empirical degree distribution
    if G is not None:
        emp_centerbins, emp_freq, emp_deg = plot_degree(G, bins=bins, display=False)

    # Plot posterior predictive intervals
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.fill_between(centerbins, quantile_freq[0, :], quantile_freq[1, :], color='lightblue', label='95% posterior predictive')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Plot the empirical degree distribution
    if G is not None:
        ax.loglog(emp_centerbins, emp_freq, '+', markersize=10,color='tab:red', label='Data')
    ax.tick_params(axis='x', labelsize=fontsize-2)
    ax.tick_params(axis='y', labelsize=fontsize-2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Degree (log scale)', fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('Frequency (log scale)', fontsize=fontsize)
    ax.set_title('Posterior predictive degree distribution', fontsize=fontsize+2)
    # Add text annotation
    #stats_text = f"Mean alpha: {ALPHA.mean():.4f}\nMean sigma: {SIGMA.mean():.4f}\nMean Dist: {Dist.mean():.4f}"
    #ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
    #    horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    plt.tight_layout()
    if legend:
        plt.legend(fontsize=fontsize)
    if plot_folder is not None:
        plot_path = os.path.join(plot_folder, f'posteriorDegree.png')
        plt.savefig(plot_path, dpi=dpi)
    plt.show()
