import numpy as np
import jraph


def to_graphs_tuple(adjacencies, size=None):
    if size is None:
        size = _nearest_power_of_2(int(adjacencies.sum()))

    num_graphs, num_variables = adjacencies.shape[:2]
    counts, sources, targets = np.nonzero(adjacencies)
    total_num_edges = counts.size

    n_node = np.full((num_graphs + 1,), num_variables, dtype=np.int32)
    n_node[-1] = 1  # Padding

    n_edge = np.bincount(counts, minlength=num_graphs + 1).astype(np.int32)
    n_edge[-1] = size - total_num_edges  # Padding

    nodes = np.tile(np.arange(num_variables, dtype=np.int32), num_graphs)
    nodes = np.append(nodes, 0)  # Padding

    edges = np.ones((size,), dtype=np.float32)

    senders = np.full((size,), num_graphs * num_variables, dtype=np.int32)
    senders[:total_num_edges] = sources + counts * num_variables

    receivers = np.full((size,), num_graphs * num_variables, dtype=np.int32)
    receivers[:total_num_edges] = targets + counts * num_variables

    globals = np.ones((num_graphs + 1,), dtype=np.float32)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=n_node,
        n_edge=n_edge,
    )


def _nearest_power_of_2(x):
    # https://stackoverflow.com/a/14267557
    return 1 if (x == 0) else (1 << (x - 1).bit_length())
