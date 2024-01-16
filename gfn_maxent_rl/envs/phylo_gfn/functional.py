import jax.numpy as jnp


def reset(batch_size, sequences):
    num_nodes = sequences.shape[0]
    max_actions = num_nodes * (num_nodes - 1) // 2

    return {
        'trees': jnp.repeat(sequences[None], batch_size, axis=0),
        'mask': jnp.ones((batch_size, max_actions), dtype=jnp.bool_),
        'type': jnp.ones((batch_size, num_nodes), dtype=jnp.int32)
    }


def step(states, actions):
    batch_size, num_nodes = states['trees'].shape[:2]
    stop_action = states['mask'].shape[1]
    arange = jnp.arange(batch_size)

    lefts, rights = jnp.triu_indices(num_nodes, k=1)
    dones = (actions == stop_action)
    actions = jnp.where(dones, 0, actions)  # To avoid overflow, will not be used

    left, right = lefts[actions], rights[actions]

    trees, types = states['trees'], states['type']

    # Merge the trees and place into "left"
    overlap = trees[arange, left] & trees[arange, right]
    union = trees[arange, left] | trees[arange, right]
    new_trees = jnp.where(overlap > 0, overlap, union)

    trees = trees.at[arange, left].set(new_trees)
    types = types.at[arange, left].set(2)

    # Remove the tree from "right"
    trees = trees.at[arange, right].set(0)
    types = types.at[arange, right].set(0)

    # Update the masks
    new_masks = states['mask']
    new_masks = jnp.logical_and(new_masks, lefts != right[:, None])
    new_masks = jnp.logical_and(new_masks, rights != right[:, None])

    return {'trees': trees, 'mask': new_masks, 'type': types}


def state_to_observation(states):
    sequences = ((states['trees'][..., None] & (1 << jnp.arange(5))) > 0)

    return {
        'sequences': sequences.astype(jnp.float32),
        'type': states['type'],
        'mask': states['mask'].astype(jnp.float32)
    }
