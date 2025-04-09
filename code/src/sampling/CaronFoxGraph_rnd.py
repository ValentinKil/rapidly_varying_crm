import numpy as np
from scipy.sparse import coo_matrix

# Todo : Improove the description of the output


def CRMtoGraph(CRM, missing_mass=0, store=False, Rapid=True):
    """
    Convert a Completely Random Measure (CRM) into a random graph following the Caron-Fox model.

    Args:
    ------
        CRM (ndarray): The Completely Random Measure, represented as an array of weights.
        missing_mass (float, optional): The total sum of the remaining weights of the CRM due to truncation. Default is 0.
        store (bool, optional): Whether to return additional information about the graph and weights. Default is False.
        Rapid (bool, optional): A flag for rapid computation (currently unused). Default is True.

    Returns:
    --------
        coo_matrix: A sparse matrix representing the undirected graph. Each entry (i, j) indicates the presence of an edge between nodes i and j.
        np.ndarray (optional): An array of weights corresponding to the nodes in the graph. Nodes with no associated weight are assigned a weight of 0.
        float (optional): The total weight of nodes not included in the graph, including the missing mass.
        coo_matrix (optional): A sparse matrix representing the directed multigraph before symmetrization.
        np.ndarray (optional): Indices of the active nodes in the original weight array that are part of the graph.

    Notes:
    -------
        - If `store` is False, only the undirected graph (coo_matrix) is returned.
        - The graph is constructed based on the Caron-Fox model, which uses the CRM weights to sample edges.

    Reference
    ----------
    [1] Caron, F., & Fox, E. B. "Sparse graphs using completely random measures". JRSSB, (2017).
    """
    # Sampling of a Poisson(W*W)
    if np.shape(CRM)[0] == 2:
        w = CRM[0, :]
    else:
        w = CRM
    N = len(w)  # Number of weights
    T = np.cumsum(w)
    cumsum_w = np.concatenate(([0], T))
    W_star = cumsum_w[-1]  # Total sum of the weights of the GG, before truncation

    if W_star == 0:
        D_star1 = 0
    else:
        # Total number of directed edges between nodes w_i before truncation
        D_star1 = np.random.poisson(W_star ** 2)

    if missing_mass <= 0 or W_star == 0:
        D_star2 = 0
    else:
        try:
            # Total number of directed edges between a node w_i and a node due to missing mass
            D_star2 = np.random.poisson(2*W_star*missing_mass)
        except ValueError:
            print('W_star:', W_star)
            print('missing_mass:', missing_mass)

    if missing_mass == 0:
        D_star3 = 0
    else:
        # Total number of directed edges between nodes due to missing mass
        D_star3 = np.random.poisson(missing_mass ** 2)

    D_star_all = D_star1 + D_star2 + D_star3  # Total number of directed edges
    N_nodes2 = D_star2  # Total number of nodes from the missing mass with a connection with a w_i
    # Total number of nodes from the missing mass with a connection with another node from the missing mass
    N_nodes3 = 2 * D_star3

    # binedges is D_star_all x 2 matrix where binedges(k, :) provides the indices (i,j) of the two nodes of directed edge k (maybe np.digitize(temp, cumsum_w))
    binedges = np.zeros((D_star_all, 2), dtype=int)
    temp1 = W_star * np.random.rand(D_star1, 2)
    binedges[:D_star1, :] = np.searchsorted(cumsum_w, temp1)
    temp2 = W_star * np.random.rand(D_star2)
    binedges[D_star1:D_star1+D_star2, 0] = np.searchsorted(cumsum_w, temp2)
    binedges[D_star1:D_star1+D_star2,
             1] = np.linspace(N, N + D_star2 - 1, D_star2, dtype=int)
    binedges[D_star1+D_star2:,
             0] = np.linspace(N + D_star2, N + D_star2 + D_star3 - 1, D_star3, dtype=int)
    binedges[D_star1+D_star2:, 1] = np.linspace(
        N + D_star2 + D_star3, N + D_star2 + 2*D_star3 - 1, D_star3, dtype=int)

    # ind, ia, ib  = np.unique(binedges, return_index = True, return_inverse = True);
    ind, ib = np.unique(binedges, return_inverse=True)
    # ind = ind.astype(int)
    Nbvertices = len(ind)

    # Construction of the graph
    ib = np.reshape(ib, (binedges.shape))
    row = ib[:, 0]  # row=np.array(ib[:,0])
    col = ib[:, 1]
    diagonal_mask = (row == col)
    data = np.array(np.ones(D_star_all))
    data[diagonal_mask] = 0
    # G is the directed multigraph
    G = coo_matrix((data, (row, col)), shape=(Nbvertices, Nbvertices))
    Gtransp = coo_matrix((data, (col, row)), shape=(Nbvertices, Nbvertices))
    Gnew = coo_matrix(G + Gtransp, dtype=bool)  # Gnew is the simple graph
    Gnew = coo_matrix(Gnew, dtype=int)

    indlog = np.zeros(w.shape, dtype=bool)
    indactivenodes_w = ind[ind <= N] - 1
    indlog[ind[ind <= N]-1] = True
    w_rem0 = sum(w[indlog == False]) + missing_mass
    # nodes for which we cannot evaluate the weight are set to 0 by default - maybe should be NaN instead
    w0 = np.zeros(Nbvertices)
    # w0[:] = np.nan # alternatively
    w0[:len(ind[ind <= N])] = w[indactivenodes_w]

    # #Compute the missing node and edges due to approximation methods
    # if missing_mass > 0:
    #     missing_edges = 2* missing_mass * (np.sum(w0) + w_rem0 + missing_mass) # expected number of missing edges
    #     missing_nodes = 2* missing_mass * np.sum(w0) +  4 * missing_mass * (w_rem0 + missing_mass) # exp. nb of missing nodes; assumes new edges are all single-edge connections, and connections within missing mass are between two different nodes
    # else :
    #     missing_edges=0
    #     missing_nodes=0

    if store:
        return Gnew, w0, w_rem0, G, indactivenodes_w  # , missing_edges, missing_nodes
    else:
        return Gnew
