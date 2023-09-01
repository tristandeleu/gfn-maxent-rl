import numpy as np
import math

from itertools import product, permutations
from scipy.special import logsumexp


def get_all_dags_compressed(num_variables):
    # Generate all the DAGs over num_variables nodes
    shape = (num_variables, num_variables)
    repeat = num_variables * (num_variables - 1) // 2

    # Generate all the possible binary codes
    codes = list(product([0, 1], repeat=repeat))
    codes = np.asarray(codes)

    # Get upper-triangular indices
    x, y = np.triu_indices(num_variables, k=1)

    # Fill the upper-triangular matrices
    trius = np.zeros((len(codes),) + shape, dtype=np.int_)
    trius[:, x, y] = codes

    # Apply permutation, and remove duplicates
    compressed_dags = set()
    for perm in permutations(range(num_variables)):
        permuted = trius[:, :, perm][:, perm, :]
        permuted = permuted.reshape(-1, num_variables ** 2)
        permuted = np.packbits(permuted, axis=1)
        compressed_dags.update(map(tuple, permuted))
    compressed_dags = sorted(list(compressed_dags))

    return np.asarray(compressed_dags)


def all_dags_batch_iterator(num_variables, batch_size=256):
    dags_compressed = get_all_dags_compressed(num_variables)
    num_batches = math.ceil(dags_compressed.shape[0] / batch_size)
    dags_compressed = np.array_split(dags_compressed, num_batches, axis=0)

    for compressed in dags_compressed:
        # Uncompress the adjacency matrices
        adjacencies = np.unpackbits(compressed, axis=1, count=num_variables ** 2)
        yield adjacencies.reshape(-1, num_variables, num_variables)


def get_exact_posterior(joint_model, nodelist, batch_size=256):
    num_variables = joint_model.num_variables
    log_posterior = {}

    for adjacencies in all_dags_batch_iterator(num_variables, batch_size=batch_size):
        log_probs = joint_model.log_prob(adjacencies)

        for adjacency, log_prob in zip(adjacencies, log_probs):
            sources, targets = np.nonzero(adjacency)
            edges = frozenset((nodelist[i], nodelist[j])
                for (i, j) in zip(sources, targets))
            log_posterior[edges] = log_prob

    # Compute the log-partition function
    log_probs = np.asarray(log_posterior.values())
    log_Z = logsumexp(log_probs)

    # Normalize the rewards
    for key, value in log_posterior.items():
        log_posterior[key] = value - log_Z

    return log_posterior
