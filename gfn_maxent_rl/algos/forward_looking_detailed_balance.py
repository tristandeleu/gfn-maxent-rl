import jax.numpy as jnp
import haiku as hk
import optax
import jax

from collections import namedtuple

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoState, AlgoParameters


DBParameters = namedtuple('DBParameters', ['policy', 'flow'])


class ForwardLookingDetailedBalance(BaseAlgorithm):
    r"""Forward-Looking Detailed Balance [1].

    Version of the Detailed Balance loss, where intermediate reward is
    available. The residual can be written as

        \Delta(s, s') = \log \frac{R(s, s')F(s')P_B(s | s')}{F(s)P_F(s' | s)}

    References
    ----------
    [1] Ling Pan, Nikolay Malkin, Dinghuai Zhang, and Yoshua Bengio. Better
        Training of GFlowNets with Local Credit and Incomplete Trajectories.
        International Conference on Machine Learning, 2023.
    """
    def __init__(self, env, policy_network, flow_network, target=None, target_kwargs={}):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.policy_network = hk.without_apply_rng(hk.transform_with_state(policy_network))
        self.flow_network = hk.without_apply_rng(hk.transform_with_state(flow_network))

    def loss(self, online_params, target_params, state, samples):
        # Get log P_F(. | s_t) for the current state
        log_pi_t, _ = self.policy_network.apply(
            online_params.policy, state.policy, samples['observation'])

        # Get log F(s_t) for the current state
        log_F_t, _ = self.flow_network.apply(
            online_params.flow, state.flow, samples['observation'])

        # Get log F(s_t+1) for the next state
        params = target_params if self.use_target else online_params
        log_F_tp1, _ = self.flow_network.apply(
            params.flow, state.flow, samples['next_observation'])

        # Compute the forward-looking detailed balance loss
        log_pF = jnp.take_along_axis(log_pi_t, samples['action'], axis=-1)
        log_pF = jnp.squeeze(log_pF, axis=-1)
        log_pB = -jnp.log(self.env.num_parents(samples['next_observation']))  # Uniform p_B

        # Recall that `samples['reward']` contains the delta-scores: log R(s_t+1) - log R(s_t)
        rewards = jnp.squeeze(samples['reward'], axis=1)

        errors = rewards + log_F_tp1 + log_pB - log_F_t - log_pF
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))

        logs = {'errors': errors, 'loss': loss}

        return (loss, logs)

    def init(self, key, normalization=1):
        subkey1, subkey2 = jax.random.split(key)

        # Initialize the network parameters (both online, and possibly target)
        policy_params, policy_state = self.policy_network.init(subkey1, self._dummy_observation)
        flow_params, flow_state = self.flow_network.init(subkey2, self._dummy_observation)
        online_params = DBParameters(policy=policy_params, flow=flow_params)

        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Set the normalization to the size of the dataset
        policy_state['~']['normalization'] = jnp.full_like(
            policy_state['~']['normalization'], normalization)
        net_state = DBParameters(policy=policy_state, flow=flow_state)

        # Initialize the state
        state = AlgoState(
            optimizer=self.optimizer.init(online_params),
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.policy_network.apply(params.policy, state.policy, observations)

        return log_pi
