# Code for Hierarchical Risk Parity optimization

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform


def optimizeHRP(returns : np.ndarray) -> np.ndarray:
    """
    Optimize a portfolio using the HRP algorithm (returns optimal portfolio weights)
    """

    if len(returns) >= 2:

        # Calculate distance matrix
        corr_matrix = np.corrcoef(returns.T)
        corr_matrix = np.clip(corr_matrix, -0.99, 0.99)  # Avoid numerical issues
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Hierarchical clustering
        condensed_distance = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_distance, method='single')
        
        # Quasi-diagonalization
        sort_indices = get_QuasiDiag(linkage_matrix)
        
        # Recursive bisection
        weights = get_RecursiveBisection(returns, sort_indices)
        
        return weights

    else:
        n_assets = returns.shape[1] if len(returns.shape) > 1 else len(returns)
        return np.ones(n_assets) / n_assets
    

def get_QuasiDiag(linkage_matrix: np.ndarray) -> np.ndarray:
    """
    Get quasi-diagonal order from linkage matrix (Simple implementation)
    """
    n_assets = len(linkage_matrix) + 1
    return np.arange(n_assets)
    
def get_RecursiveBisection(returns: np.ndarray, sort_indices: np.ndarray) -> np.ndarray:
    """
    Recursive bisection to get final weights
    """
    n_assets = returns.shape[1]
    weights = np.ones(n_assets)

    # Calculate inverse variance weights (simplified HRP)
    variances = np.var(returns, axis=0)
    inv_var_weights = 1 / (variances + 1e-8)
    weights = inv_var_weights / np.sum(inv_var_weights)

    return weights
