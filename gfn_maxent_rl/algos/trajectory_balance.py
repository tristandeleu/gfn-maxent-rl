import jax.numpy as jnp
import jax
import optax
import warnings

from gfn_maxent_rl.algos.base import GFNBaseAlgorithm


class GFNTrajectoryBalance(GFNBaseAlgorithm):
    def __init__(self, env, network, update_target_every=0):
        if update_target_every != 0:
            warnings.warn('No target network used in GFNTrajectoryBalance, but '
                f'`update_target_every={update_target_every}`. Setting '
                '`update_target_every=0`.')
        super().__init__(env, network, update_target_every=0)

    def loss(self, online_params, _, state, samples):
        # Get log P_F(. | G_t) for full trajectory
        v_model = jax.vmap(self.network.apply, in_axes=(None, None, 0))
        log_pi, _ = v_model(online_params.network, state, samples['observations'])

        # Mask the log-probabilities, based on the sequence lengths
        seq_masks = (jnp.arange(log_pi.shape[1]) <= samples['lengths'][:, None])
        log_pi = jnp.where(seq_masks[..., None], log_pi, 0.)

        # Compute the forward log_probabilities
        log_pF = jnp.take_along_axis(log_pi, samples['actions'][..., None], axis=-1)
        log_pF = jnp.sum(log_pF, axis=(1, 2))

        # Compute the backward log-probabilities (fixed P_B)
        log_pB = -jax.lax.lgamma(samples['lengths'] + 1.)  # -log(n!)

        # Compute the log-rewards, based on the delta-scores
        log_rewards = jnp.where(seq_masks, samples['rewards'], 0.)
        log_rewards = jnp.sum(log_rewards, axis=1)

        errors = online_params.log_Z + log_pF - log_rewards - log_pB
        loss = jnp.mean(optax.l2_loss(errors))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)
