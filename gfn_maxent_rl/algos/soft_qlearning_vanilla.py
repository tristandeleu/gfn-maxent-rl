import jax.numpy as jnp
import haiku as hk
import optax

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoParameters, AlgoState


class SoftQLearningVanilla(BaseAlgorithm):
    def __init__(self, network, update_target_every=0):
        super().__init__(update_target_every=update_target_every)
        self.network = hk.without_apply_rng(hk.transform_with_state(network))

    def loss(self, online_params, target_params, state, samples):
        batch_size = samples['mask'].shape[0]
        masks_continue = samples['mask'].reshape(batch_size, -1)
        mask_stop = jnp.ones((batch_size, 1), dtype=masks_continue.dtype)
        masks = jnp.concatenate((masks_continue, mask_stop), axis=-1)

        # Get Q(G_t, .) for the current graph
        Q_t, _ = self.network.apply(
            online_params, state, samples['graph'], samples['mask'])

        # Get Q(G_t+1, .) for the next graph
        params = target_params if self.use_target else online_params
        Q_tp1, _ = self.network.apply(
            params, state, samples['next_graph'], samples['next_mask'])
        Q_tp1 = jnp.where(masks == 1., Q_tp1, 0.)
        V_tp1 = Q_tp1.sum(axis=-1)

        # Compute the (modified) detailed balance loss
        old_Q = jnp.take_along_axis(Q_t, samples['action'], axis=-1)
        old_Q = jnp.squeeze(old_Q, axis=-1)

        rewards = jnp.squeeze(samples['reward'], axis=1)
        errors = (rewards + V_tp1 - old_Q)
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))  # TODO: Modify delta

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)

    def init(self, key, samples, normalization=1):
        # Initialize the network parameters (both online, and possibly target)
        online_params, net_state = self.network.init(key, samples['graph'], samples['mask'])
        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # # Set the normalization to the size of the dataset
        # net_state['~']['normalization'] = jnp.full_like(
        #     net_state['~']['normalization'], normalization)

        # Initialize the state
        state = AlgoState(
            optimizer=self.optimizer.init(online_params),
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.network.apply(
            params,
            state,
            observations['graph'],
            observations['mask']
        )
        return log_pi
