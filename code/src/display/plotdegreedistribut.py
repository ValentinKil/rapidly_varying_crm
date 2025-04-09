
import numpy as np
import matplotlib.pyplot as plt

basecol = np.array([139, 0, 0]) / 255


def plot_degree(G, linespec='o', step=1, bins=16,fontsize=22,display=True):
    """
    plot_degree plots the degree of each node for the observed adjacency matrix G

    Aargs:
    -------
      - G: observed binary adjacency matrix (scipy.sparse.coo_matrix)
      - linespec: line specification (determines line type, marker symbol, and color of the plotted lines)
      - step: step size for the logarithmic bin edges in the pdf of the degree distribution
      - fontsize: font size for the plot labels

    Return:
    -------
      - fig: loglog degree distribution figure
      - ax: axis of the figure
      - centerbins: bin centers
      - freq: frequencies for the counts of the degrees in each bin
    """

    G = G.tocoo()  # Ensure G is in COO format
    G.data = np.nan_to_num(G.data, nan=0.5)  # fill missing by 0.5
    deg = np.array(G.sum(axis=0)).flatten()


    # Uses logarithmic binning to get a less noisy estimate of the
    # pdf of the degree distribution
    edgebins = 2.0 ** np.arange(0, bins + step, step)
    sizebins = edgebins[1:] - edgebins[:-1]

    centerbins = edgebins[:-1] #+ sizebins / 2
    counts, _ = np.histogram(deg, bins=edgebins)
    freq = counts / sizebins / G.shape[0]

    if display : 
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.loglog(centerbins, freq, linespec, alpha = 0.8,color="c")
        plt.xlabel('Degree', fontsize=fontsize)
        plt.ylabel('Distribution', fontsize=fontsize)
        #plt.grid(True, which="both", ls="--")

        #plt.show()
        return fig, ax, centerbins, freq, deg
    else : 
        return centerbins, freq, deg
