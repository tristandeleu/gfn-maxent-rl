import jax.numpy as jnp
import haiku as hk
import optax
import jax

from collections import namedtuple

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoParameters, AlgoState


SACParameters = namedtuple('SACParameters', ['actor', 'critic1', 'critic2'])

class SAC(BaseAlgorithm):
    def __init__(self, env, actor_network, critic_network, target=None, target_kwargs={}):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.actor_network = hk.without_apply_rng(hk.transform_with_state(actor_network))
        self.critic_network = hk.without_apply_rng(hk.transform_with_state(critic_network))

    def loss(self, online_params, target_params, state, samples):
        masks = samples['observation']['mask']
        batch_size = masks.shape[0]
        masks_continue = masks.reshape(batch_size, -1)
        mask_stop = jnp.ones((batch_size, 1), dtype=masks_continue.dtype)
        masks = jnp.concatenate((masks_continue, mask_stop), axis=-1)

        # actor network
        log_pi, _ = self.actor_network.apply(
            online_params.actor, state.actor, samples['observation'])

        # Get Q(G_t, .) for the current graph
        Q1_t, _ = self.critic_network.apply( 
            online_params.critic1, state.critic1, samples['observation'])
        Q2_t, _ = self.critic_network.apply(
            online_params.critic2, state.critic2, samples['observation'])

        # Get Q(G_t+1, .) for the next graph
        params = target_params if self.use_target else online_params
        Q1_tp1, _ = self.critic_network.apply(
            params.critic1, state.critic1, samples['next_observation'])
        Q2_tp1, _ = self.critic_network.apply(
            params.critic2, state.critic2, samples['next_observation'])

        # critic loss
        log_pi_nograd = jax.lax.stop_gradient(log_pi)
        min_Q_tp1 = jnp.exp(log_pi_nograd) * (jnp.minimum(Q1_tp1, Q2_tp1) - log_pi_nograd)
        min_Q_tp1 = jnp.where(masks == 1., min_Q_tp1, 0.)
        V_tp1 = min_Q_tp1.sum(axis=-1)

        # use Q-values only for the taken actions
        old_Q1 = jnp.take_along_axis(Q1_t, samples['action'], axis=-1)
        old_Q1 = jnp.squeeze(old_Q1, axis=-1)
        old_Q2 = jnp.take_along_axis(Q2_t, samples['action'], axis=-1)
        old_Q2 = jnp.squeeze(old_Q2, axis=-1)

        # critic loss
        rewards = jnp.squeeze(samples['reward'], axis=-1)
        old_Q1_errors = (rewards + V_tp1 - old_Q1)
        old_Q1_loss = optax.huber_loss(old_Q1_errors, delta=1.)  # TODO: Modify delta
        old_Q2_errors = (rewards + V_tp1 - old_Q2)
        old_Q2_loss = optax.huber_loss(old_Q2_errors, delta=1.)  # TODO: Modify delta

        critic_loss = old_Q1_loss + old_Q2_loss

        # actor loss
        min_Q_t = jax.lax.stop_gradient(jnp.minimum(Q1_t, Q2_t))
        actor_loss = jnp.exp(log_pi) * (log_pi - min_Q_t)
        actor_loss = jnp.where(masks == 1., actor_loss, 0.)
        actor_loss = jnp.sum(actor_loss, axis=-1)

        loss = jnp.mean(critic_loss + actor_loss)

        logs = {'loss': loss, 'critic_loss': critic_loss, 'actor_loss': actor_loss}
        return (loss, logs)

    def init(self, key, samples, normalization=1):
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)

        # Initialize the network parameters (both online, and possibly target)
        actor_params, actor_state = self.actor_network.init(subkey1, samples['observation'])
        critic1_params, critic1_state = self.critic_network.init(subkey2, samples['observation'])
        critic2_params, critic2_state = self.critic_network.init(subkey3, samples['observation'])
        online_params = SACParameters(actor=actor_params, critic1=critic1_params, critic2=critic2_params)

        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Set the normalization to the size of the dataset
        actor_state['~']['normalization'] = jnp.full_like(
            actor_state['~']['normalization'], normalization)
        net_state = SACParameters(actor=actor_state, critic1=critic1_state, critic2=critic2_state)

        # Initialize the state
        state = AlgoState(
            optimizer=self.optimizer.init(online_params),
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.actor_network.apply(params.actor, state.actor, observations)
        return log_pi
