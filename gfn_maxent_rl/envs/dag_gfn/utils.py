import jax.numpy as jnp
import jraph


def to_graphs_tuple(adjacencies, size):
    num_graphs, num_variables = adjacencies.shape[:2]
    counts, senders, receivers = jnp.nonzero(adjacencies, size=size, fill_value=-1)

    n_node = jnp.full((num_graphs,), num_variables, dtype=jnp.int_)
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
