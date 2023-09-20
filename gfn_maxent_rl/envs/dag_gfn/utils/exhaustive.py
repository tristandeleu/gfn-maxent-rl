import numpy as np
import math

from itertools import product, permutations


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


def get_all_dags_keys(dags_compressed, num_variables, batch_size=256):
    num_batches = math.ceil(dags_compressed.shape[0] / batch_size)
    dags_compressed = np.array_split(dags_compressed, num_batches, axis=0)
    keys = []

    for compressed in dags_compressed:
        # Uncompress the adjacency matrices
        adjacencies = np.unpackbits(compressed, axis=1, count=num_variables ** 2)
        adjacencies = adjacencies.reshape(-1, num_variables, num_variables)

        for adjacency in adjacencies:
            keys.append(frozenset(zip(*np.nonzero(adjacency))))

    return keys
