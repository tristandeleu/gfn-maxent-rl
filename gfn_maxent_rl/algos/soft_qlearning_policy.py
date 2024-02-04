import jax.numpy as jnp
import haiku as hk
import optax

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoParameters, AlgoState


class SoftQLearningPolicy(BaseAlgorithm):
    r"""Soft Q-Learning with policy parametrization.

    This is a formulation of SQL adapted to the case where all the states of
    the Markov Decision Process are terminating. This is parametrized by
    a policy instead of a Q-function. The residual can be written as

        \Delta(s, s') = \log \pi(s' | s) - \log \pi(s_f | s) + \log \pi(s_f | s') - r(s, s')
    """
    def __init__(self, env, network, target=None, target_kwargs={}):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.network = hk.without_apply_rng(hk.transform_with_state(network))

    def loss(self, online_params, target_params, state, samples):
        # Get log pi(. | s_t) for the current state
        log_pi_t, _ = self.network.apply(
            online_params, state, samples['observation'])

        # Get log pi(. | s_t+1) for the next state
        params = target_params if self.use_target else online_params
        log_pi_tp1, _ = self.network.apply(
            params, state, samples['next_observation'])

        old_log_pi = jnp.take_along_axis(log_pi_t, samples['action'], axis=-1)
        old_log_pi = jnp.squeeze(old_log_pi, axis=-1)

        rewards = jnp.squeeze(samples['reward'], axis=1)
        errors = (rewards + log_pi_t[:, -1] - log_pi_tp1[:, -1] - old_log_pi)
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)
    
    def init(self, key, normalization=1):
        # Initialize the network parameters (both online, and possibly target)
        online_params, net_state = self.network.init(key, self._dummy_observation)
        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Set the normalization to the size of the dataset
        net_state['~']['normalization'] = jnp.full_like(
            net_state['~']['normalization'], normalization)

        # Initialize the state
        state = AlgoState(
            optimizer=self.optimizer.init(online_params),
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.network.apply(params, state, observations)
        return log_pi
