import jax.numpy as jnp
import jax


def batch_random_choice(key, log_probs, axis=-1):
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    # Sample from a Gumbel distribution
    z = jax.random.gumbel(key, shape=log_probs.shape)
    # Use the Gumbel-max trick
    return jnp.argmax(log_probs + z, axis=axis)
