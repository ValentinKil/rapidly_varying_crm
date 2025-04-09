
import numpy as np
from scipy.sparse import coo_matrix

def pSampling(G, p,notP=True):
    def pSampling(G, p, notP=True):
        """
        Perform  p Sampling on a sparse matrix and optionally return its complement.
        This function takes a sparse matrix `G` in COO format and performs probabilistic sampling 
        to retain a subset of its rows and columns based on a probability `p`. It can also return 
        the complement of the sampled matrix if `notP` is set to True.
        
        Args:
        -----------
        G : scipy.sparse.coo_matrix
            The input sparse matrix in COO format to be sampled.
        p : float
            The probability of retaining each row and column in the matrix.
        notP : bool, optional
            If True, the function also computes and returns the complement of the sampled matrix.
            Default is True.
            
        Returns:
        --------
        final_subG : scipy.sparse.coo_matrix
            The sampled submatrix with rows and columns retained based on the probability `p`.
        not_final_subG : scipy.sparse.coo_matrix, optional
            The complement of the sampled submatrix, returned only if `notP` is True.
            
        References:
        -----------
        [1] Veitch and Roy, "Sampling and estimation for (sparse) exchangeable graphs", The Annals of Statistics, (2019), https://arxiv.org/abs/1611.00843
        """
    # G is assumed to be a scipy.sparse.coo_matrix
    n = G.shape[0]
    tokeep = np.random.rand(n) < p  # Boolean mask for rows and columns to keep
    if notP:
        not_tokeep = ~tokeep  # Boolean mask for rows and columns to remove
    
    # Create masks for rows and columns that are kept or discarded
    row_mask = tokeep[G.row]
    col_mask = tokeep[G.col]
    mask = row_mask & col_mask  # Only keep entries where both row and col are in 'tokeep'
    
    if notP:
        not_row_mask = not_tokeep[G.row]
        not_col_mask = not_tokeep[G.col]
        not_mask = not_row_mask & not_col_mask  # Only remove entries where both row and col are in 'not_tokeep'

    # Filter and create submatrices
    subn = np.sum(tokeep)
    if notP:
        not_subn = np.sum(not_tokeep)
    
    # Remap the indices for the kept rows and columns
    idx_map = np.where(tokeep)[0]
    
    if notP:
        not_idx_map = np.where(not_tokeep)[0]
    
    subG = coo_matrix(
        (G.data[mask], (np.searchsorted(idx_map, G.row[mask]), np.searchsorted(idx_map, G.col[mask]))),
        shape=(subn, subn)
    )
    
    if notP:
        not_subG = coo_matrix(
        (G.data[not_mask], (np.searchsorted(not_idx_map, G.row[not_mask]), np.searchsorted(not_idx_map, G.col[not_mask]))),
        shape=(not_subn, not_subn)
    )

    # Remove empty rows and columns in subG and not_subG
    rows_with_nonzero = np.unique(subG.row)
    cols_with_nonzero = np.unique(subG.col)
    
    if notP:
        not_rows_with_nonzero = np.unique(not_subG.row)
        not_cols_with_nonzero = np.unique(not_subG.col)

    # Apply compact index mapping directly
    row_map = {old_idx: new_idx for new_idx, old_idx in enumerate(rows_with_nonzero)}
    col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(cols_with_nonzero)}

    if notP:
        not_row_map = {old_idx: new_idx for new_idx, old_idx in enumerate(not_rows_with_nonzero)}
        not_col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(not_cols_with_nonzero)}

    # Apply the new row/column mappings to the sub-matrices
    new_rows = np.array([row_map[r] for r in subG.row])
    new_cols = np.array([col_map[c] for c in subG.col])
    
    if notP:
        not_new_rows = np.array([not_row_map[r] for r in not_subG.row])
        not_new_cols = np.array([not_col_map[c] for c in not_subG.col])

    # Create the final reduced matrices
    final_subG = coo_matrix(
        (subG.data, (new_rows, new_cols)),
        shape=(len(rows_with_nonzero), len(cols_with_nonzero))
    )
    if notP:
        not_final_subG = coo_matrix(
        (not_subG.data, (not_new_rows, not_new_cols)),
        shape=(len(not_rows_with_nonzero), len(not_cols_with_nonzero))
    )
    if notP:
        return final_subG, not_final_subG
    else:
        return final_subG
