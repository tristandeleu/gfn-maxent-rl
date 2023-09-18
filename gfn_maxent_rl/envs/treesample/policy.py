import jax.numpy as jnp


def uniform_log_policy(masks, num_categories):
    num_valid_actions = num_categories * jnp.sum(masks, axis=1, keepdims=True)
    log_policy = jnp.where(masks == 1., -jnp.log(num_valid_actions), -jnp.inf)
    return jnp.repeat(
        log_policy,
        repeats=num_categories,
        total_repeat_length=num_categories * masks.shape[1]
    )


def action_mask(masks, num_categories):
    masks = jnp.repeat(
        masks,
        repeats=num_categories,
        total_repeat_length=num_categories * masks.shape[1]
    )
    return masks.astype(jnp.bool_)
