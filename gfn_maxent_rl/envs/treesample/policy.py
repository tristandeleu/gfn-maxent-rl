import jax.numpy as jnp
import haiku as hk
import jax


def log_policy(logits, masks, num_categories):
    logp_stop = jnp.where(jnp.any(masks, axis=1, keepdims=True), -jnp.inf, 0.)
    logits = jnp.concatenate((logits, logp_stop), axis=1)

    # Mask out invalid actions
    action_masks = action_mask(masks, num_categories)
    logits = jnp.where(action_masks, logits, -jnp.inf)

    return jax.nn.log_softmax(logits, axis=1)


def uniform_log_policy(masks, num_categories):
    num_valid_actions = num_categories * jnp.sum(masks, axis=1, keepdims=True)
    log_policy = jnp.where(masks == 1., -jnp.log(num_valid_actions), -jnp.inf)
    logp_continue = jnp.repeat(
        log_policy,
        repeats=num_categories,
        axis=1,
        total_repeat_length=num_categories * masks.shape[1]
    )
    logp_stop = jnp.where(jnp.any(masks, axis=1, keepdims=True), -jnp.inf, 0.)
    return jnp.concatenate((logp_continue, logp_stop), axis=1)


def action_mask(masks, num_categories):
    masks_continue = jnp.repeat(
        masks,
        repeats=num_categories,
        axis=1,
        total_repeat_length=num_categories * masks.shape[1]
    ).astype(jnp.bool_)
    masks_stop = jnp.logical_not(jnp.any(masks, axis=1, keepdims=True))
    return jnp.concatenate((masks_continue, masks_stop), axis=1)


def policy_network(num_categories):
    def network(observations):
        num_variables = observations['variables'].shape[1]
        output_size = num_variables * num_categories

        # First layer of the MLP
        hiddens = hk.Embed(
            num_categories + 1,
            embed_dim=256
        )(observations['variables'] + 1)
        hiddens = jnp.sum(hiddens, axis=1)
        hiddens = jax.nn.leaky_relu(hiddens)

        # Rest of the MLP
        logits = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(hiddens)

        # Mask out the invalid actions
        norm = hk.get_state('normalization', (), init=jnp.ones)
        return log_policy(logits * norm, observations['mask'], num_categories)

    return network


def q_network(num_categories):
    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = num_variables * num_categories

        # First layer of the MLP
        hiddens = hk.Embed(
            num_categories + 1,
            embed_dim=256
        )(observations['variables'] + 1)
        hiddens = jnp.sum(hiddens, axis=1)
        hiddens = jax.nn.leaky_relu(hiddens)

        # Rest of the MLP
        q_values_continue = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(hiddens)

        # Value of the stop action is 0
        q_value_stop = jnp.zeros((batch_size, 1), dtype=q_values_continue.dtype)

        return jnp.concatenate((q_values_continue, q_value_stop), axis=1)

    return network


def f_network(num_categories):
    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = 1

        one_hots = jax.nn.one_hot(observations['variables']+1, num_categories+1)
        one_hots = one_hots.reshape(batch_size, num_variables*(num_categories+1))
        
        # First layer of the MLP
        # hiddens = hk.Embed(
        #     num_categories + 1,
        #     embed_dim=256, lookup_style='ONE_HOT'
        # )(one_hots.astype(int))

        # Rest of the MLP
        f_values_continue = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(one_hots)

        return jnp.squeeze(f_values_continue, axis=-1)

    return network

