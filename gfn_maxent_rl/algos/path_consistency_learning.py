import jax.numpy as jnp
import jax
import optax
import warnings

from gfn_maxent_rl.algos.base import GFNBaseAlgorithm


class PathConsistencyLearning(GFNBaseAlgorithm):
    r"""Path Consistency Learning objective [1].

    PCL objective applied to complete trajectories only. The residual can be written as

        \Delta(\tau) = -V(s_0) + V(s_T) + \sum_{t=0}^{T-1} (r(s_t, s_t+1) - \log \pi(s_t+1 | s_t))

    References
    ----------
    [1] Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging
        the Gap Between Value and Policy Based Reinforcement Learning. Advances
        in Neural Information Processing Systems, 2017.
    """
    def __init__(self, env, network, target=None, target_kwargs={}):
        if target is not None:
            warnings.warn('No target network used in PCL, but '
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

        # Compute the log-rewards, based on the delta-scores
        log_rewards = jnp.where(seq_masks, samples['rewards'], 0.)
        log_rewards = jnp.sum(log_rewards, axis=1)

        # PCL loss (log_Z = log V(s_0))
        errors = online_params.log_Z + log_pF - log_rewards
        loss = jnp.mean(optax.l2_loss(errors))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)
