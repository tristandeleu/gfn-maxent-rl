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


def batch_sequences_to_graphs_tuple(num_variables, actions, lengths):
    batch_size, max_length = actions.shape
    max_edges_in_sequence = max_length * (max_length - 1) // 2

    n_node = np.full((batch_size, max_length + 1), num_variables, dtype=np.int32)
    n_node[:, -1] = 1  # Padding

    nodes = np.zeros((batch_size, num_variables * max_length + 1), dtype=np.int32)
    nodes[:, :-1] = np.tile(np.arange(num_variables), (batch_size, max_length))

    n_edge = np.tile(np.arange(max_length + 1), (batch_size, 1))
    n_edge = np.where(n_edge <= lengths[:, None], n_edge, 0)
    n_edge[:, -1] = max_edges_in_sequence - np.sum(n_edge, axis=1)  # Padding

    edges = np.ones((batch_size, max_edges_in_sequence), dtype=np.float32)

    offsets, indices = np.tril_indices(max_length - 1)
    senders, receivers = divmod(actions[:, indices], num_variables)

    senders = np.where(offsets < lengths[:, None],
        senders + (offsets + 1) * num_variables, max_length * num_variables)
    receivers = np.where(offsets < lengths[:, None],
        receivers + (offsets + 1) * num_variables, max_length * num_variables)

    globals = np.ones((batch_size, max_length + 1), dtype=np.float32)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=n_node,
        n_edge=n_edge
    )
