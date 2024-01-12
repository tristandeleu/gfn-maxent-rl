import jax.numpy as jnp
import haiku as hk
import jax

import dataclasses


def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


@dataclasses.dataclass
class Transformer(hk.Module):
    """A transformer stack.
    inspired from:
    https://github.com/google-deepmind/dm-haiku/blob/6339353cab51d3d71f172d1bbe2d216c33ce09a4/examples/transformer/model.py#L41
    """

    num_heads: int = 8  # Number of attention heads.
    num_layers: int = 6  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int = 32  # Size of the attention (key, query, value) vectors.
    dropout_rate: float = 0.1  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.

    # name: Optional[str] = None  # Optional identifier for the module.

    def __call__(
            self,
            embeddings: jax.Array,  # [B, T, D]
            mask=None,  # [B, T]
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        # _, seq_len, model_size = embeddings.shape
        model_size = embeddings.shape[-1]

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

            if mask is not None:
                h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask[:, None, None, :])
            elif mask is None:
                h_attn = attn_block(h_norm, h_norm, h_norm)  # , mask=mask
            else:
                raise ValueError("mask should be None or a boolean array")
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
