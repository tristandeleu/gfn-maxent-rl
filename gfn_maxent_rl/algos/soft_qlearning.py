import jax.numpy as jnp
import haiku as hk
import optax
import jax

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoParameters, AlgoState


class SoftQLearning(BaseAlgorithm):
    r"""Soft Q-Learning objective [1].

    The residual can be written as

        \Delta(s, s') = Q(s, s') - (r(s, s') + logsumexp(Q(s', s'')))

    References
    ----------
    [1] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine.
        Reinforcement Learning with Deep Energy-Based Policies. International
        Conference on Machine Learning, 2017.
    """
    def __init__(self, env, network, target=None, target_kwargs={}):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.network = hk.without_apply_rng(hk.transform_with_state(network))

    def loss(self, online_params, target_params, state, samples):
        # Get Q(s_t, .) for the current state
        Q_t, _ = self.network.apply(
            online_params, state, samples['observation'])

        # Get Q(G_s+1, .) for the next state
        params = target_params if self.use_target else online_params
        Q_tp1, _ = self.network.apply(
            params, state, samples['next_observation'])
        V_tp1 = jax.nn.logsumexp(Q_tp1, axis=1)

        # Compute the (modified) detailed balance loss
        old_Q = jnp.take_along_axis(Q_t, samples['action'], axis=-1)
        old_Q = jnp.squeeze(old_Q, axis=-1)

        rewards = jnp.squeeze(samples['reward'], axis=1)
        errors = (rewards + V_tp1 - old_Q)
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)

    def init(self, key, normalization=1):
        # Initialize the network parameters (both online, and possibly target)
        online_params, net_state = self.network.init(key, self._dummy_observation)
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
        log_pi = jax.nn.log_softmax(q_values, axis=-1)
        return log_pi
