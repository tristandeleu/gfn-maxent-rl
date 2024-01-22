import numpy as np
import haiku as hk

from typing import Mapping


def save(filename, trees):
    data = {}
    for prefix, tree in trees.items():
        data[prefix] = tree

    np.savez(filename, **data)


def load(filename):
    data = {}
    f = open(filename, 'rb') if isinstance(filename, str) else filename
    results = np.load(f, allow_pickle=True)

    for key in results.files:
        data[key] = results[key]


    if isinstance(filename, str):
        f.close()
    del results
    # returns dictionary of the networks
    return data