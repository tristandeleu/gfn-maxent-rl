import numpy as np
import pandas as pd
import networkx as nx
import jax.numpy as jnp
import pandas as pd
import pickle

from numpy.random import default_rng


def sample_from_linear_gaussian(graph, num_samples, rng=default_rng()):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if graph.graph.get('type', '') != 'linear-gaussian':
        raise ValueError('The graph is not a Linear Gaussian Bayesian Network.')

    samples = pd.DataFrame(columns=list(graph.nodes()))
    for node in nx.topological_sort(graph):
        attrs = graph.nodes[node]
        if attrs['parents']:
            values = np.vstack([samples[parent] for parent in attrs['parents']])
            mean = attrs['bias'] + np.dot(attrs['cpd'], values)
            samples[node] = rng.normal(mean, attrs['obs_scale'])
        else:
            samples[node] = rng.normal(
                attrs['bias'],
                attrs['obs_scale'],
                size=(num_samples,)
            )
    return samples


def load_artifact_continuous(artifact_dir):
    with open(artifact_dir / 'graph.pkl', 'rb') as f:
        graph = pickle.load(f)

    train = pd.read_csv(artifact_dir / 'train_data.csv', index_col=0, header=0)
    train = jnp.asarray(train)

    valid = pd.read_csv(artifact_dir / 'valid_data.csv', index_col=0, header=0)
    valid = jnp.asarray(valid)

    return train, valid, graph
