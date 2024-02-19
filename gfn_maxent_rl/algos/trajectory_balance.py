import jax.numpy as jnp
import jax
import optax
import warnings

from gfn_maxent_rl.algos.base import GFNBaseAlgorithm


class TrajectoryBalance(GFNBaseAlgorithm):
    r"""Trajectory Balance loss [1].

    The residual can be written as

        \Delta(\tau) = \log \frac{R(s_T)\prod_{t=0}^{T-1}P_B(s_t | s_t+1)}{Z\prod_{t=0}^{T}P_F(s_t+1 | s_t)}

    References
    ----------
    [1] Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio.
        Trajectory balance: Improved credit assignment in GFlowNets. Advances
        in Neural Information Processing Systems, 2022.
    """
    def __init__(self, env, network, target=None, target_kwargs={}):
        if target is not None:
            warnings.warn('No target network used in GFNTrajectoryBalance, but '
                f'`target={target}`. Setting `target=None`.')
        super().__init__(env, network, target=None, target_kwargs={})

    def loss(self, online_params, _, state, samples):
        # Get log P_F(. | s_t) for full trajectory
        v_model = jax.vmap(self.network.apply, in_axes=(None, None, 0))
        log_pi, _ = v_model(online_params.network, state, samples['observations'])

        # Mask the log-probabilities, based on the sequence lengths
        seq_masks = (jnp.arange(log_pi.shape[1]) <= samples['lengths'][:, None])
        log_pi = jnp.where(seq_masks[..., None], log_pi, 0.)

        # Compute the forward log_probabilities
        log_pF = jnp.take_along_axis(log_pi, samples['actions'][..., None], axis=-1)
        log_pF = jnp.sum(log_pF, axis=(1, 2))

        # Compute the backward log-probabilities (fixed P_B)
        num_parents = self.env.num_parents(samples['observations'])
        # Default to 0 for log p_B if num_parents == 0 (e.g., padding)
        num_parents = jnp.maximum(num_parents, 1)
        log_pB = -jnp.sum(jnp.log(num_parents), axis=-1)

        # Compute the log-rewards, based on the delta-scores
        log_rewards = jnp.where(seq_masks, samples['rewards'], 0.)
        log_rewards = jnp.sum(log_rewards, axis=1)

        errors = online_params.log_Z + log_pF - log_rewards - log_pB
        loss = jnp.mean(optax.l2_loss(errors))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)
