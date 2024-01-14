import jax.numpy as jnp
import jax
import jraph


def reset(batch_size, num_variables):
    # See gfn_maxent_rl/envs/dag_gfn/env.py
    shape = (batch_size, num_variables, num_variables)
    closure_T = jnp.eye(num_variables, dtype=jnp.bool_)
    return {
        'adjacency': jnp.zeros(shape, dtype=jnp.bool_),
        'closure_T': jnp.tile(closure_T, (batch_size, 1, 1))
    }


def step(states, actions):
    # See gfn_maxent_rl/envs/dag_gfn/env.py
    batch_size, num_variables = states['adjacency'].shape[:2]
    arange = jnp.arange(batch_size)

    sources, targets = jnp.divmod(actions, num_variables)
    dones = (sources == num_variables)
    sources = jnp.where(dones, 0, sources)  # To avoid overflow, will not be used

    # Update adjacency matrices
    new_adjacency = states['adjacency'].at[arange, sources, targets].set(True)
    null_adjacency = jnp.zeros_like(states['adjacency'])
    adjacency = jnp.where(dones[:, None, None], null_adjacency, new_adjacency)

    # Update transitive closure
    source_rows = jnp.expand_dims(states['closure_T'][arange, sources, :], axis=1)
    target_cols = jnp.expand_dims(states['closure_T'][arange, :, targets], axis=2)
    update = jnp.logical_and(source_rows, target_cols)  # Outer product
    new_closure_T = jnp.logical_or(states['closure_T'], update)
    null_closure_T = jnp.ones_like(states['closure_T'])
    closure_T = jnp.where(dones[:, None, None], null_closure_T, new_closure_T)

    return {'adjacency': adjacency, 'closure_T': closure_T}


def trajectories_to_graphs_tuple(trajectories, num_variables):
    batch_size, max_edges = trajectories.shape
    masks = (trajectories >= 0)  # Valid entries in "trajectories"

    nodes = jnp.tile(jnp.arange(num_variables), batch_size)
    nodes = jnp.append(nodes, 0)  # Padding

    n_node = jnp.full((batch_size,), num_variables, dtype=jnp.int32)
    n_node = jnp.append(n_node, 1)  # Padding

    max_num_edges = batch_size * max_edges
    n_edge = jnp.sum(masks, axis=1)
    n_edge = jnp.append(n_edge, max_num_edges - jnp.sum(n_edge))  # Padding

    sources = jnp.where(masks, trajectories // num_variables, -1)
    targets = jnp.where(masks, trajectories % num_variables, -1)

    # Compute common indices
    indices = jnp.cumsum(masks, dtype=jnp.int32)
    indices = jnp.maximum(indices - 1, 0)
    indices = jnp.where(masks, indices.reshape(masks.shape), -1)

    edges = jnp.ones((max_num_edges,), dtype=jnp.float32)
    globals_ = jnp.ones((batch_size + 1,), dtype=jnp.float32)

    num_nodes = batch_size * num_variables
    offsets = jnp.arange(batch_size) * num_variables
    offsets = jnp.expand_dims(offsets, axis=1)

    senders = jnp.full((max_num_edges + 1,), num_nodes, dtype=jnp.int32)
    senders = senders.at[indices].set(sources + offsets)
    receivers = jnp.full((max_num_edges + 1,), num_nodes, dtype=jnp.int32)
    receivers = receivers.at[indices].set(targets + offsets)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers[:-1],
        senders=senders[:-1],
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
    )


def state_to_observation(states, trajectories):
    num_variables = states['adjacency'].shape[1]
    return {
        'adjacency': states['adjacency'].astype(jnp.float32),
        'mask': 1. - jnp.asarray(states['adjacency'] + states['closure_T'], dtype=jnp.float32),
        'graph': trajectories_to_graphs_tuple(trajectories, num_variables)
    }
