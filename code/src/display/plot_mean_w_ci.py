
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_mean_w_ci(G, objmcmc, w_true=None, nburn=0, fontsize=14, plot_folder=None,dpi=300,figsize=(8, 6)):
    """
    Plot credible intervals for the sociability parameters (weights) of nodes in a graph.
    This function visualizes the credible intervals for the weights of nodes in a graph, 
    sorted by their degree. It generates two plots:
    1. Credible intervals for high-degree nodes.
    2. Credible intervals for low-degree nodes.
    Args:
    -----------
    G : numpy.ndarray
        Adjacency matrix of the graph (shape: [n, n]).
    objmcmc : object
        MCMC object containing posterior samples of the weights. It should have the 
        structure `objmcmc.samples[j]["w"]` for each chain `j` and settings in 
        `objmcmc.settings["nchains"]`.
    w_true : numpy.ndarray, optional
        True weights of the nodes (shape: [n,]). If provided, the true values will be 
        plotted alongside the credible intervals.
    nburn : int, optional
        Number of burn-in samples to discard from the MCMC chains. Default is 0.
    fontsize : int, optional
        Font size for plot labels and titles. Default is 14.
    plot_folder : str, optional
        Path to the folder where the plots will be saved. If None, the plots are not saved. 
        Default is None.
        
    Notes:
    ------
    - High-degree nodes are sorted in descending order of their degree, and the top 50 
      nodes (or fewer if the graph has fewer nodes) are plotted.
    - Low-degree nodes are selected randomly from nodes with degree 1. If `w_true` is 
      provided, only nodes with positive true weights are considered.
    - The credible intervals are computed as the 2.5th and 97.5th percentiles of the 
      posterior samples.
    - For low-degree nodes, the logarithm of the weights is plotted.
    
    Outputs:
    --------
    - Displays two plots:
        1. Credible intervals for high-degree nodes.
        2. Credible intervals for low-degree nodes (logarithmic scale).
    - If `plot_folder` is specified, the plots are saved as:
        - `w_highdeg.png` for high-degree nodes.
        - `w_lowdeg.png` for low-degree nodes.
    """

    
    degree = np.array(np.sum(G, axis=0) + np.sum(G, axis=1).T).squeeze()/2
    ind = np.argsort(degree)[::-1]  # Sorted by descending degree
    nnodes=G.shape[0]
    # mean_w_true = np.mean(w_true)

    # High degree nodes
    plt.figure(dpi=dpi,figsize=figsize)
    plt.title('Credible intervals - high degree nodes',fontsize=fontsize+2)
    plt.xlabel('Index of node (sorted by dec. degree)', fontsize=fontsize)
    plt.ylabel('Sociability parameters', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    first = range(min(nnodes, 50))
    i=first[0]
    samples_w = np.array([objmcmc.samples[j]["w"][nburn:,ind[i]] for j in range(objmcmc.settings["nchains"])])
    q = np.quantile(samples_w.flatten(), [0.025, 0.975])
    plt.plot([i, i], q, 'tab:red', linewidth=3, marker='.', markersize=8, label='Credible interval')
    for i in first[1:]:
        samples_w = np.array([objmcmc.samples[j]["w"][nburn:,ind[i]] for j in range(objmcmc.settings["nchains"])])
        q = np.quantile(samples_w.flatten(), [0.025, 0.975])
        plt.plot([i, i], q, 'tab:red', linewidth=3, marker='.', markersize=8)
    
    if w_true is not None:
        plt.plot(first, w_true[ind[first]], marker='x',color='tab:green',linestyle='None', linewidth=2,label='True value')
    plt.xlim(-0.9, min(nnodes, 50) + 0.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=fontsize-2, loc='upper right')
    plt.tight_layout()
    if plot_folder is not None:
        plot_path = os.path.join(plot_folder, f'w_highdeg.png')
        plt.savefig(plot_path, dpi=dpi)
    plt.show()
    plt.close()

 

    # Low degree nodes
    plt.figure(dpi=dpi,figsize=figsize)
    plt.title('Credible intervals - low degree nodes',fontsize=fontsize+2)
    plt.xlabel('Nodes', fontsize=fontsize)
    plt.ylabel('Log sociability parameters', fontsize=fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    

    if w_true is not None:
        ind2 = np.where((w_true>0) & (degree==1))[0]
    else :
        ind2 = np.where((degree==1))[0]
    last = np.random.choice(ind2, 50)# range(max(0, max_node2 - 50), max_node2)
    for j in range(len(last)):
        i=last[j]
        samples_w = np.array([objmcmc.samples[j]["w"][nburn:,ind[i]] for j in range(objmcmc.settings["nchains"])])
        q = np.quantile(np.log(samples_w.flatten()), [0.025, 0.975])
        plt.plot([j, j], q, 'tab:red', linewidth=3, marker='.', markersize=8)
        if w_true is not None:
            plt.plot(j, np.log(w_true[i]), marker='x',color='tab:green',linestyle='None', linewidth=2)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plot_folder is not None:
        plot_path = os.path.join(plot_folder, f'w_lowdeg.png')
        plt.savefig(plot_path, dpi=dpi)
    plt.show()
    plt.close()
 
