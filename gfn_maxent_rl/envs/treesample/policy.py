import jax.numpy as jnp
import haiku as hk
import jax

import dataclasses


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
    # def network(observations):
    #     batch_size, num_variables = observations['variables'].shape
    #     output_size = 1
    #
    #     one_hots = jax.nn.one_hot(observations['variables']+1, num_categories+1)
    #     one_hots = one_hots.reshape(batch_size, num_variables*(num_categories+1))
    #
    #     # First layer of the MLP
    #     # hiddens = hk.Embed(
    #     #     num_categories + 1,
    #     #     embed_dim=256, lookup_style='ONE_HOT'
    #     # )(one_hots.astype(int))
    #
    #     # Rest of the MLP
    #     f_values_continue = hk.nets.MLP(
    #         (256, 256, output_size),
    #         activation=jax.nn.leaky_relu
    #     )(one_hots)
    #
    #     return jnp.squeeze(f_values_continue, axis=-1)
    #
    # return network

    def network(observations):
        batch_size, num_variables = observations['variables'].shape
        output_size = 1
        transformer = Transformer()
        embed_dim = 256
        seq_len = 5
        model_size = 128

        one_hots = jax.nn.one_hot(observations['variables'] + 1, num_categories + 1)
        token_embeddings = one_hots.reshape(batch_size, num_variables*(num_categories+1))

        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embeddings = hk.Embed(
            num_categories + 1,
            embed_dim=256,
            w_init=embed_init
        )(token_embeddings.astype(int))

        # positional_embeddings = jnp.zeros_like(token_embeddings)
        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [30, 256], init=embed_init) # [6, 256]
        # import pdb; pdb.set_trace()

        input_embeddings = token_embeddings + positional_embeddings

        embeddings = transformer(input_embeddings)  # , input_mask)

        # Pass the output of Transformer to a one-layer MLP
        # hiddens = hk.Embed(
        #     num_categories + 1,
        #     output_size
        # )(embeddings)

        out = hk.Linear(output_size)(embeddings)

        return out

    return network


@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack.
  inspired from:
  https://github.com/google-deepmind/dm-haiku/blob/6339353cab51d3d71f172d1bbe2d216c33ce09a4/examples/transformer/model.py#L41
  """

  num_heads: int = 8  # Number of attention heads.
  num_layers: int = 6  # Number of transformer (attention + MLP) layers to stack.
  attn_size: int = 32 # Size of the attention (key, query, value) vectors.
  dropout_rate: float = 0.1  # Probability with which to apply dropout.
  widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
  # name: Optional[str] = None  # Optional identifier for the module.

  def __call__(
      self,
      embeddings: jax.Array,  # [B, T, D]
      # mask: jax.Array,  # [B, T]
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""

    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    _, seq_len, model_size = embeddings.shape

    # Compute causal mask for autoregressive sequence modelling.
    # mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
    # causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]
    # mask = mask * causal_mask  # [B, H=1, T, T]

    h = embeddings
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.attn_size,
          model_size=model_size,
          w_init=initializer,
      )
      h_norm = _layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm)  # , mask=mask
      # h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
      h = h + h_attn

      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = _layer_norm(h)
      h_dense = dense_block(h_norm)
      # h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
      h = h + h_dense

    return _layer_norm(h)


def _layer_norm(x: jax.Array) -> jax.Array:
  """Applies a unique LayerNorm to `x` with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)

