import networkx as nx
import scipy.sparse as sp
import numpy as np

def coo_to_networkx(coo_matrix):
    """
    Converts a graph represented as a COO matrix into a NetworkX graph.
    
    Parameters:
    - coo_matrix (scipy.sparse.coo_matrix): The adjacency matrix in COO format.
    
    Returns:
    - G (networkx.Graph): The corresponding NetworkX graph.
    """
    
    if not sp.isspmatrix_coo(coo_matrix):
        raise ValueError("Input matrix must be in COO format.")
    
    # Create an empty undirected graph
    G = nx.Graph()
    
    # Add edges from COO matrix
    rows, cols, data = coo_matrix.row, coo_matrix.col, coo_matrix.data
    for u, v, w in zip(rows, cols, data):
        G.add_edge(u, v, weight=w)
    
    return G