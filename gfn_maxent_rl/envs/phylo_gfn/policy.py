import jax.numpy as jnp
import jax


def uniform_log_policy(masks):
    logits = jnp.where(action_mask(masks), 0., -jnp.inf)
    return jax.nn.log_softmax(logits, axis=-1)

def action_mask(masks):
    action_masks = masks.astype(jnp.bool_)
    is_terminal = jnp.any(action_masks, axis=-1, keepdims=True)
    return jnp.concatenate((action_masks, ~is_terminal), axis=-1)
