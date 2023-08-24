import jax.numpy as jnp
import optax

from gfn_maxent_rl.algos.base import BaseAlgorithm


class GFNDetailedBalance(BaseAlgorithm):
    def loss(self, online_params, target_params, samples):
        # Get log P_F(. | G_t) for the current graph
        log_pi_t = self.network.apply(
            online_params, samples['graph'], samples['mask'])

        # Get log P_F(. | G_t+1) for the next graph
        params = target_params if self.update_target_every > 0 else online_params
        log_pi_tp1 = self.network.apply(
            params, samples['next_graph'], samples['next_mask'])

        # Compute the (modified) detailed balance loss
        log_pF = jnp.take_along_axis(log_pi_t, samples['action'], axis=-1)
        log_pF = jnp.squeeze(log_pF, axis=-1)
        log_pB = -jnp.log1p(samples['graph'].n_edge[:-1])  # [:-1] -> Remove padding

        # Recall that `samples['reward']` contains the delta-scores: log R(G') - log R(G)
        errors = (samples['reward'] + log_pB + log_pi_t[:, -1] - log_pF - log_pi_tp1[:, -1])
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))  # TODO: Modify delta

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)
