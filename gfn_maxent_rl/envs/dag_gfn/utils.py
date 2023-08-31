import jax.numpy as jnp
import jraph


def to_graphs_tuple(adjacencies, size):
    num_graphs, num_variables = adjacencies.shape[:2]
    counts, senders, receivers = jnp.nonzero(adjacencies, size=size, fill_value=-1)

    n_node = jnp.full((num_graphs,), num_variables, dtype=jnp.int32)
    n_node = jnp.append(n_node, 1)  # Padding

    counts = jnp.where(counts < 0, num_graphs, counts)
    n_edge = jnp.bincount(counts, length=num_graphs + 1)

    nodes = jnp.tile(jnp.arange(num_variables), num_graphs)
    nodes = jnp.append(nodes, 0)  # Padding

    edges = jnp.ones_like(senders)

    senders = jnp.where(senders < 0, 0, senders)
    senders = senders + counts * num_variables  # Offset

    receivers = jnp.where(receivers < 0, 0, receivers)
    receivers = receivers + counts * num_variables  # Offset

    globals = jnp.ones((num_graphs + 1,), dtype=jnp.float32)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=n_node,
        n_edge=n_edge,
    )


def trajectory_to_graphs_tuple(actions, length, num_variables):
    max_length = actions.shape[0]
    max_edges_in_sequence = max_length * (max_length - 1) // 2

    # Number of nodes:
    # The number of nodes in each graph is equal to the number of variables,
    # except the last graph at index "max_length" of each sequence, which is a
    # graph with a single node, used for padding.
    n_node = jnp.full((max_length,), num_variables, dtype=jnp.int32)
    n_node = jnp.append(n_node, 1)  # Padding

    # Nodes:
    # The nodes are encoded as their integer index in [0, num_variables - 1].
    # The last graph of each sequence has a single node with index "0".
    nodes = jnp.tile(jnp.arange(num_variables), max_length)
    nodes = jnp.append(nodes, 0)  # Padding

    # Number of edges:
    # The number of edges increases by one for each graph in the sequence,
    # starting from the empty graph, up to the stop action. Then we pad the
    # sequence with empty graphs (where n_edge = 0).
    # The last graph (with a single node) contains as many edges as there are to
    # sum the total number of edges in all graphs of the episodes to "max_edge_in_sequence".
    n_edge = jnp.arange(max_length + 1)
    n_edge = jnp.where(n_edge <= length, n_edge, 0)
    n_edge[-1] = max_edges_in_sequence - jnp.sum(n_edge, axis=1)  # Padding

    # Edges:
    # All the edges are encoded with the same embedding (with index "1"). There
    # are a total of "max_edges_in_sequence" edges in all the graphs of a
    # sequence (including the padding graph).
    edges = jnp.ones((max_edges_in_sequence,), dtype=jnp.int32)

    indices, offsets = jnp.tril_indices(max_length - 1)
    offsets = offsets + 1
    senders, receivers = jnp.divmod(actions[indices], num_variables)

    senders = jnp.where(offsets < length, senders + offsets * num_variables,
                        max_length * num_variables)
    receivers = jnp.where(offsets < length, receivers + offsets * num_variables,
                          max_length * num_variables)

    globals = jnp.ones((max_length + 1,), dtype=jnp.float32)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals,
        n_node=n_node,
        n_edge=n_edge
    )
