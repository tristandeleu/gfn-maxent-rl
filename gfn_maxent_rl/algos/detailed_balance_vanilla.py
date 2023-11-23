import jax.numpy as jnp
import haiku as hk
import optax
import jax

from collections import namedtuple

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoState
from functools import partial
# from gfn_maxent_rl.envs.dag_gfn import DAGEnvironment

DBVParameters = namedtuple('DBVParameters', ['policy', 'flow'])
AlgoParameters = namedtuple('AlgoParameters', ['online', 'target'])


class GFNDetailedBalanceVanilla(BaseAlgorithm):
    def __init__(self, env, policy_network, flow_network, target=None, target_kwargs={}):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.policy_network = hk.without_apply_rng(hk.transform_with_state(policy_network))
        self.flow_network = hk.without_apply_rng(hk.transform_with_state(flow_network))

    def _apply_flow(self, params, state, observations):
        F, _ = self.flow_network.apply(params, state, observations)
        # Q2, _ = self.critic_network.apply(params[1], state[1], observations)
        # return (Q1, Q2)
        return F

    def loss(self, online_params, target_params, state, samples):
        """
        P_F: Forward policy
        P_B: Backward policy
        F: Flow
        """
        action_masks = self.env.action_mask(samples['next_observation'])

        # Get log P_F(. | G_t) for the current graph
        log_pi_t, _ = self.policy_network.apply(
            online_params.policy, state.policy, samples['observation'])

        # Get log F(G_t, .) for the current graph
        log_F_t = self._apply_flow(
            online_params.flow, state.flow, samples['observation'])

        # # Get log P_F(. | G_t+1) for the next graph
        # params = target_params if self.use_target else online_params
        # log_pi_tp1, _ = self.policy_network.apply(
        #     params.policy, state.policy, samples['next_observation'])

        # Get log F(G_t+1, .) for the next graph
        params = target_params if self.use_target else online_params
        log_F_tp1 = self._apply_flow(
            params.flow, state.flow, samples['next_observation'])

        # Compute the detailed balance (vanilla) loss
        log_pF = jnp.take_along_axis(log_pi_t, samples['action'], axis=-1)
        log_pF = jnp.squeeze(log_pF, axis=-1)
        log_pB = -jnp.log(self.env.num_parents(samples['next_observation']))  # Uniform p_B

        # Recall that `samples['reward']` contains the delta-scores: log R(G') - log R(G)
        rewards = jnp.squeeze(samples['reward'], axis=1)

        errors = rewards + log_F_tp1 + log_pB - log_F_t - log_pF
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)

    def init(self, key, samples, normalization=1):
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)

        # Initialize the network parameters (both online, and possibly target)
        policy_params, policy_state = self.policy_network.init(subkey1, samples['observation'])
        flow_params, flow_state = self.flow_network.init(subkey2, samples['observation'])
        online_params = DBVParameters(policy=policy_params, flow=flow_params)

        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Set the normalization to the size of the dataset
        policy_state['~']['normalization'] = jnp.full_like(
            policy_state['~']['normalization'], normalization)
        net_state = DBVParameters(policy=policy_state, flow=flow_state)

        # Initialize the state of the optimizers
        opt_state = self.optimizer.init(online_params)

        # Initialize the state
        state = AlgoState(
            optimizer=opt_state,
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.policy_network.apply(params.policy, state.policy, observations)

        return log_pi
