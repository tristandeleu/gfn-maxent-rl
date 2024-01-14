import jax.numpy as jnp


def reset(batch_size, num_variables):
    return jnp.full((batch_size, num_variables), -1, dtype=jnp.int32)


def step(states, actions, num_categories):
    batch_size, num_variables = states.shape
    arange = jnp.arange(batch_size)

    indices, values = jnp.divmod(actions, num_categories)
    dones = (indices == num_variables)
    indices = jnp.where(dones, 0, indices)  # To avoid overflow, will not be used

    new_states = states.at[arange, indices].set(values)
    return jnp.where(dones[:, None], states, new_states)

def state_to_observation(states):
    return {
        'variables': states,
        'mask': (states == -1).astype(jnp.int32)
    }
