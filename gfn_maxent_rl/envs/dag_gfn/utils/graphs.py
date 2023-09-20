import numpy as np


def transitive_closure(adjacencies):
    closure = adjacencies.astype(np.bool_)
    num_nodes = adjacencies.shape[1]

    for i in range(num_nodes):
        outer_product = np.logical_and(closure[:, :, None, i], closure[:, i, None, :])
        closure = np.logical_or(closure, outer_product)
    
    closure = np.logical_or(closure, np.eye(num_nodes, dtype=np.bool_))  # Convention
    return closure


def compute_masks(adjacencies):
    adjacencies = adjacencies.astype(np.bool_)
    closure_T = transitive_closure(adjacencies)
    closure_T = closure_T.transpose((0, 2, 1))
    return np.logical_not(np.logical_or(adjacencies, closure_T))
