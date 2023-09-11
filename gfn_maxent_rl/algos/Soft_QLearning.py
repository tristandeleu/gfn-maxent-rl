import jax.numpy as jnp
import optax

from gfn_maxent_rl.algos.base import BaseAlgorithm


class SoftQLearning(BaseAlgorithm):
    def loss(self, online_params, target_params, state, samples):
        # Get log P_F(. | G_t) for the current graph
        Q_t, _ = self.network.apply(
            online_params, state, samples['graph'], samples['mask'])

        # Get log P_F(. | G_t+1) for the next graph
        params = target_params if self.update_target_every > 0 else online_params
        Q_tp1, _ = self.network.apply(
            params, state, samples['next_graph'], samples['next_mask'])

        # Compute the (modified) detailed balance loss
        old_Q = jnp.take_along_axis(Q_t, samples['action'], axis=-1)
        old_Q = jnp.squeeze(old_Q, axis=-1)
        # log_pB = -jnp.log1p(samples['graph'].n_edge[:-1])  # [:-1] -> Remove padding

        # Recall that `samples['reward']` contains the delta-scores: log R(G') - log R(G)
        rewards = jnp.squeeze(samples['reward'], axis=1)
        errors = (rewards + Q_t[:, -1] - Q_tp1[:, -1] - old_Q)
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))  # TODO: Modify delta

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)

