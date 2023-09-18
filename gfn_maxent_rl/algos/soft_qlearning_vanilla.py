import jax.numpy as jnp
import haiku as hk
import optax
import jax

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoParameters, AlgoState


class SoftQLearningVanilla(BaseAlgorithm):
    def __init__(self, env, network, update_target_every=0):
        super().__init__(env, update_target_every=update_target_every)
        self.network = hk.without_apply_rng(hk.transform_with_state(network))

    def loss(self, online_params, target_params, state, samples):
        action_masks = self.env.action_mask(samples['observation'])

        # Get Q(G_t, .) for the current graph
        Q_t, _ = self.network.apply(
            online_params, state, samples['observation'])

        # Get Q(G_t+1, .) for the next graph
        params = target_params if self.use_target else online_params
        Q_tp1, _ = self.network.apply(
            params, state, samples['next_observation'])
        Q_tp1 = jnp.where(action_masks, Q_tp1, 0.)
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
        online_params, net_state = self.network.init(key, samples['observation'])
        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Initialize the state
        state = AlgoState(
            optimizer=self.optimizer.init(online_params),
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        q_values, _ = self.network.apply(params, state, observations)

        # Mask invalid actions
        action_masks = self.env.action_mask(observations)
        logits = jnp.where(action_masks, q_values, -jnp.inf)
        log_pi = jax.nn.log_softmax(logits, axis=-1)

        return log_pi
