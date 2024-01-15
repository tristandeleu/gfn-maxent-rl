import jax.numpy as jnp
import haiku as hk
import jax

from gfn_maxent_rl.nets.transformer import Transformer


def uniform_log_policy(masks):
    logits = jnp.where(action_mask(masks), 0., -jnp.inf)
    return jax.nn.log_softmax(logits, axis=-1)


def action_mask(masks):
    action_masks = masks.astype(jnp.bool_)
    is_terminal = jnp.any(action_masks, axis=-1, keepdims=True)
    return jnp.concatenate((action_masks, ~is_terminal), axis=-1)


def log_policy(logits, masks):
    logp_stop = jnp.where(jnp.any(masks, axis=1, keepdims=True), -jnp.inf, 0.)
    logits = jnp.concatenate((logits, logp_stop), axis=1)

    # Mask out invalid actions
    logits = jnp.where(action_mask(masks), logits, -jnp.inf)

    return jax.nn.log_softmax(logits, axis=1)


def policy_network_mlp():
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        encoding = hk.nets.MLP(
            (256, 256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        # Pairwise combination
        row, col = jnp.triu_indices(num_nodes, k=1)
        encodings_combination = encoding[:, row] + encoding[:, col]

        # Tree topology MLP
        logits = hk.nets.MLP(
            (256, 256, 1),
            activation=jax.nn.leaky_relu
        )(encodings_combination)
        logits = jnp.squeeze(logits, axis=-1)

        # Mask out the invalid actions
        norm = hk.get_state('normalization', (), init=jnp.ones)
        return log_policy(logits * norm, observations['mask'])

    return network


def policy_network_transformer():
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embeddings = hk.nets.MLP(
            (256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        # padding mask
        batch_padding_mask = (observations['type'] > 0).astype(jnp.float32)

        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [27, 128], init=embed_init)

        input_embeddings = token_embeddings + positional_embeddings

        encoding = Transformer()(input_embeddings, batch_padding_mask)


        # Pairwise combination
        row, col = jnp.triu_indices(num_nodes, k=1)
        encodings_combination = encoding[:, row] + encoding[:, col]

        # Tree topology MLP
        logits = hk.nets.MLP(
            (256, 256, 1),
            activation=jax.nn.leaky_relu
        )(encodings_combination)
        logits = jnp.squeeze(logits, axis=-1)


        # # Edge lengths MLP
        # edge_lengths = hk.nets.MLP(
        #     (256, 256, output_size),
        #     activation=jax.nn.leaky_relu
        # )(tree_topology)

        norm = hk.get_state('normalization', (), init=jnp.ones)
        return log_policy(logits * norm, observations['mask'])

    return network


def q_network_mlp():
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        encoding = hk.nets.MLP(
            (256, 256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        # Pairwise combination
        row, col = jnp.triu_indices(num_nodes, k=1)
        encodings_combination = encoding[:, row] + encoding[:, col]

        # Tree topology MLP
        logits = hk.nets.MLP(
            (256, 256, 1),
            activation=jax.nn.leaky_relu
        )(encodings_combination)
        q_values_continue = jnp.squeeze(logits, axis=-1)

        # Value of the stop action is 0
        q_value_stop = jnp.zeros((batch_size, 1), dtype=q_values_continue.dtype)
        outputs = jnp.concatenate((q_values_continue, q_value_stop), axis=1)

        return outputs

    return network


def q_network_transformer():
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embeddings = hk.nets.MLP(
            (256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        # padding mask
        batch_padding_mask = (observations['type'] > 0).astype(jnp.float32)

        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [27, 128], init=embed_init)

        input_embeddings = token_embeddings + positional_embeddings

        encoding = Transformer()(input_embeddings, batch_padding_mask)

        # Pairwise combination
        row, col = jnp.triu_indices(num_nodes, k=1)
        encodings_combination = encoding[:, row] + encoding[:, col]

        # Tree topology MLP
        logits = hk.nets.MLP(
            (256, 256, 1),
            activation=jax.nn.leaky_relu
        )(encodings_combination)
        q_values_continue = jnp.squeeze(logits, axis=-1)

        # Value of the stop action is 0
        q_value_stop = jnp.zeros((batch_size, 1), dtype=q_values_continue.dtype)
        outputs = jnp.concatenate((q_values_continue, q_value_stop), axis=1)

        return outputs

    return network


def f_network_mlp():
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        encoding = hk.nets.MLP(
            (256, 256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        encoding = encoding.reshape(batch_size, num_nodes * 128)
        outputs = hk.Linear(output_size)(encoding)
        outputs = jnp.squeeze(outputs, axis=-1)

        # Set the flow at terminating states to 0
        # /!\ This is assuming that the terminal state is the *only* child of
        # any terminating state, which is true for the TreeSample environments.
        is_intermediate = jnp.any(observations['mask'], axis=-1)
        outputs = jnp.where(is_intermediate, outputs, 0.)
        return outputs

    return network


def f_network_transformer():
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape
        output_size = 1

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embeddings = hk.nets.MLP(
            (256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        # padding mask
        batch_padding_mask = (observations['type'] > 0).astype(jnp.float32)

        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [27, 128], init=embed_init)

        input_embeddings = token_embeddings + positional_embeddings

        encoding = Transformer()(input_embeddings, batch_padding_mask)
        encoding = encoding.reshape(batch_size, num_nodes * 128)
        outputs = hk.Linear(output_size)(encoding)
        outputs = jnp.squeeze(outputs, axis=-1)

        # Set the flow at terminating states to 0
        # /!\ This is assuming that the terminal state is the *only* child of
        # any terminating state, which is true for the TreeSample environments.
        is_intermediate = jnp.any(observations['mask'], axis=-1)
        outputs = jnp.where(is_intermediate, outputs, 0.)
        return outputs

    return network
