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
    # action_masks = action_mask(masks)
    logits = jnp.where(action_mask(masks), logits, -jnp.inf)

    return jax.nn.log_softmax(logits, axis=1)


# def uniform_log_policy(masks, num_categories):
#     num_valid_actions = num_categories * jnp.sum(masks, axis=1, keepdims=True)
#     log_policy = jnp.where(masks == 1., -jnp.log(num_valid_actions), -jnp.inf)
#     logp_continue = jnp.repeat(
#         log_policy,
#         repeats=num_categories,
#         axis=1,
#         total_repeat_length=num_categories * masks.shape[1]
#     )
#     logp_stop = jnp.where(jnp.any(masks, axis=1, keepdims=True), -jnp.inf, 0.)
#     return jnp.concatenate((logp_continue, logp_stop), axis=1)


# def action_mask(masks, num_categories):
#     masks_continue = jnp.repeat(
#         masks,
#         repeats=num_categories,
#         axis=1,
#         total_repeat_length=num_categories * masks.shape[1]
#     ).astype(jnp.bool_)
#     masks_stop = jnp.logical_not(jnp.any(masks, axis=1, keepdims=True))
#     return jnp.concatenate((masks_continue, masks_stop), axis=1)


def policy_network_mlp(num_categories):
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape
        output_size = num_nodes * num_categories

        one_hots = jax.nn.one_hot(observations['variables'] + 1, num_categories + 1)
        one_hots = one_hots.reshape(batch_size, num_nodes * (num_categories + 1))

        logits = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(one_hots)

        # Mask out the invalid actions
        norm = hk.get_state('normalization', (), init=jnp.ones)
        return log_policy(logits * norm, observations['mask'], num_categories)

    return network


def policy_network_transformer(num_categories):
    def network(observations):
        batch_size, num_nodes, sequence_length, _ = observations['sequences'].shape # [batch, num_nodes, sequence_length, 5]
        output_size = num_categories

        input = observations['sequences'].reshape(batch_size, num_nodes, -1)
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        # token_embeddings = hk.Embed(
        #     2,
        #     embed_dim=128,
        #     w_init=embed_init
        # )(observations['sequences'].astype(int))
        token_embeddings = hk.nets.MLP(
            (256, 128),
            activation=jax.nn.leaky_relu
        )(input)

        # token_embeddings = token_embeddings.reshape(batch_size, num_nodes, sequence_length * 1)

        # padding mask
        batch_padding_mask = (observations['type'] > 0).astype(jnp.float32) #TODO: use it later in Transformer


        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [27, 128], init=embed_init)

        input_embeddings = token_embeddings + positional_embeddings

        encoding = Transformer()(input_embeddings, batch_padding_mask) #[b, num_nodes, 128]

        # logits = hk.Linear(output_size)(encoding)
        # logits = logits.reshape(batch_size, num_nodes * output_size)

        # Pairwise combination
        row, col = jnp.triu_indices(num_nodes, k=1)
        encodings_combination = encoding[:, row] + encoding[:, col]


        # Tree topology MLP
        logits = hk.nets.MLP(
            (256, 256, 1),
            activation=jax.nn.leaky_relu
        )(encodings_combination)
        logits = jnp.squeeze(logits, axis=-1)

        # tree_topology = jax.nn.log_softmax(tree_topology, axis=-1)


        # # Edge lengths MLP
        # edge_lengths = hk.nets.MLP(
        #     (256, 256, output_size),
        #     activation=jax.nn.leaky_relu
        # )(tree_topology)
        # edge_lengths = jax.nn.log_softmax(edge_lengths, axis=-1)

        norm = hk.get_state('normalization', (), init=jnp.ones)
        # import pdb; pdb.set_trace()
        return log_policy(logits * norm, observations['mask'])

    return network


def q_network_mlp(num_categories):
    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = num_variables * num_categories

        one_hots = jax.nn.one_hot(observations['variables'] + 1, num_categories + 1)
        one_hots = one_hots.reshape(batch_size, num_variables * (num_categories + 1))

        q_values_continue = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(one_hots)

        # Value of the stop action is 0
        q_value_stop = jnp.zeros((batch_size, 1), dtype=q_values_continue.dtype)
        outputs = jnp.concatenate((q_values_continue, q_value_stop), axis=1)

        return outputs

    return network


def q_network_transformer(num_categories):
    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = num_categories

        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embeddings = hk.Embed(
            num_categories + 1,
            embed_dim=256,
            w_init=embed_init
        )(observations['variables'] + 1)

        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [num_variables, 256], init=embed_init)

        input_embeddings = token_embeddings + positional_embeddings

        embeddings = Transformer()(input_embeddings)
        q_values_continue = hk.Linear(output_size)(embeddings)
        q_values_continue = q_values_continue.reshape(batch_size, num_variables * num_categories)  # [batch, num_variables * num_categories]

        # Value of the stop action is 0
        q_value_stop = jnp.zeros((batch_size, 1), dtype=q_values_continue.dtype)
        outputs = jnp.concatenate((q_values_continue, q_value_stop), axis=1)

        return outputs

    return network


def f_network_mlp(num_categories):
    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = 1

        one_hots = jax.nn.one_hot(observations['variables'] + 1, num_categories + 1)
        one_hots = one_hots.reshape(batch_size, num_variables * (num_categories + 1))

        outputs = hk.nets.MLP(
            (256, 256, output_size),
            activation=jax.nn.leaky_relu
        )(one_hots)

        outputs = jnp.squeeze(outputs, axis=-1)
        # Set the flow at terminating states to 0
        # /!\ This is assuming that the terminal state is the *only* child of
        # any terminating state, which is true for the TreeSample environments.
        is_intermediate = jnp.any(observations['mask'], axis=-1)
        outputs = jnp.where(is_intermediate, outputs, 0.)
        return outputs

    return network


def f_network_transformer(num_categories):
    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = 1
        embed_dim = 256

        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embeddings = hk.Embed(
            num_categories + 1,
            embed_dim=embed_dim,
            w_init=embed_init
        )(observations['variables'] + 1)

        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [num_variables, embed_dim], init=embed_init)

        input_embeddings = token_embeddings + positional_embeddings

        embeddings = Transformer()(input_embeddings)
        embeddings = embeddings.reshape(batch_size, num_variables * embed_dim)
        outputs = hk.Linear(output_size)(embeddings)

        outputs = jnp.squeeze(outputs, axis=-1)
        # Set the flow at terminating states to 0
        # /!\ This is assuming that the terminal state is the *only* child of
        # any terminating state, which is true for the TreeSample environments.
        is_intermediate = jnp.any(observations['mask'], axis=-1)
        outputs = jnp.where(is_intermediate, outputs, 0.)
        return outputs

    return network
