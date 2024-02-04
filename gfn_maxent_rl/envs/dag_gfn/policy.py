import jax.numpy as jnp
import jax
import haiku as hk

from gfn_maxent_rl.nets.gnn import GNNBackbone


def log_policy(logits, stop, masks):
    masks = masks.reshape(logits.shape)
    masked_logits = jnp.where(masks == 1., logits, -jnp.inf)
    can_continue = jnp.any(masks, axis=-1, keepdims=True)

    logp_continue = (jax.nn.log_sigmoid(-stop)
        + jax.nn.log_softmax(masked_logits, axis=-1))
    logp_stop = jax.nn.log_sigmoid(stop)

    # In case there is no valid action other than stop
    logp_continue = jnp.where(can_continue, logp_continue, -jnp.inf)
    logp_stop = logp_stop * can_continue

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)
    num_edges = jnp.sum(masks, axis=-1, keepdims=True)

    logp_stop = -jnp.log1p(num_edges)
    logp_continue = jnp.where(masks == 1., logp_stop, -jnp.inf)

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def action_mask(masks):
    masks_continue = masks.reshape(masks.shape[0], -1)
    mask_stop = jnp.ones((masks.shape[0], 1), dtype=masks.dtype)

    return jnp.concatenate((masks_continue, mask_stop), axis=-1, dtype=jnp.bool_)


def policy_network(observations):
    batch_size = observations['mask'].shape[0]
    features = GNNBackbone(num_layers=1, name='gnn')(
        observations['graph'], observations['mask'])

    senders = hk.nets.MLP([128, 128], name='senders')(features.nodes)
    receivers = hk.nets.MLP([128, 128], name='receivers')(features.nodes)

    logits = jax.lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
    logits = logits.reshape(batch_size, -1)
    stop = hk.nets.MLP([128, 1], name='stop')(features.globals)

    norm = hk.get_state('normalization', (), init=jnp.ones)
    return log_policy(logits * norm, stop * norm, observations['mask'])


def q_network(observations):
    batch_size = observations['mask'].shape[0]
    features = GNNBackbone(num_layers=1, name='gnn')(
        observations['graph'], observations['mask'])

    senders = hk.nets.MLP([128, 128], name='senders')(features.nodes)
    receivers = hk.nets.MLP([128, 128], name='receivers')(features.nodes)

    q_values = jax.lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
    q_values = q_values.reshape(batch_size, -1)

    # Mask the Q-values
    masks_continue = observations['mask'].reshape(batch_size, -1)
    q_values = jnp.where(masks_continue, q_values, -jnp.inf)

    q_value_stop = jnp.zeros((batch_size, 1), dtype=q_values.dtype)
    return jnp.concatenate((q_values, q_value_stop), axis=-1)


def f_network(observations):
    features = GNNBackbone(num_layers=1, name='gnn')(
        observations['graph'], observations['mask'])

    outputs = hk.nets.MLP([128, 1], name='flow')(features.globals)

    return jnp.squeeze(outputs, axis=-1)
