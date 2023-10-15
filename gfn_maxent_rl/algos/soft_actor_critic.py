import jax.numpy as jnp
import haiku as hk
import optax
import jax

from collections import namedtuple

from gfn_maxent_rl.algos.base import BaseAlgorithm, AlgoParameters, AlgoState
from functools import partial

SACParameters = namedtuple('SACParameters', ['actor', 'critic'])

class SAC(BaseAlgorithm):
    def __init__(self, env, actor_network, critic_network, target=None, target_kwargs={}, policy_frequency=1):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.actor_network = hk.without_apply_rng(hk.transform_with_state(actor_network))
        self.critic_network = hk.without_apply_rng(hk.transform_with_state(critic_network))
        self.policy_frequency = policy_frequency

    def _apply_critic(self, params, state, observations):
        Q1, _ = self.critic_network.apply(params[0], state[0], observations)
        Q2, _ = self.critic_network.apply(params[1], state[1], observations)
        return (Q1, Q2)

    def actor_loss(self, actor_online_params, critic_online_params, state, samples):
        action_masks_t = self.env.action_mask(samples['observation'])

        # Actor network
        log_pi_t, _ = self.actor_network.apply(
            actor_online_params, state.actor, samples['observation'])
        
        # Get Q(G_t, .) for the current graph
        Q1_t, Q2_t = self._apply_critic(critic_online_params, state.critic, samples['observation'])

        min_Q_t = jax.lax.stop_gradient(jnp.minimum(Q1_t, Q2_t))
        pi_t = jnp.where(action_masks_t, jnp.exp(log_pi_t), 0.)
        actor_loss = pi_t * (log_pi_t - min_Q_t)
        actor_loss = jnp.where(action_masks_t, actor_loss, 0.)
        actor_loss = jnp.sum(actor_loss, axis=-1)

        loss = jnp.mean(actor_loss)
        logs = {'actor_loss': actor_loss}
        return (loss, logs)

    def critic_loss(self, critic_online_params, actor_online_params, target_params, state, samples):
        action_masks_tp1 = self.env.action_mask(samples['next_observation'])

        # Actor network
        log_pi_tp1, _ = self.actor_network.apply(
            actor_online_params, state.actor, samples['next_observation'])

        # Get Q(G_t, .) for the current graph
        Q1_t, Q2_t = self._apply_critic(critic_online_params, state.critic, samples['observation'])

        # Get Q(G_t+1, .) for the next graph
        params = target_params.critic if self.use_target else critic_online_params
        Q1_tp1, Q2_tp1 = self._apply_critic(params, state.critic, samples['next_observation'])

        # Critic loss
        log_pi_tp1 = jax.lax.stop_gradient(log_pi_tp1)
        min_Q_tp1 = jnp.exp(log_pi_tp1) * (jnp.minimum(Q1_tp1, Q2_tp1) - log_pi_tp1)
        min_Q_tp1 = jnp.where(action_masks_tp1, min_Q_tp1, 0.)
        V_tp1 = min_Q_tp1.sum(axis=-1)

        # Use Q-values only for the taken actions
        old_Q1 = jnp.take_along_axis(Q1_t, samples['action'], axis=-1)
        old_Q1 = jnp.squeeze(old_Q1, axis=-1)
        old_Q2 = jnp.take_along_axis(Q2_t, samples['action'], axis=-1)
        old_Q2 = jnp.squeeze(old_Q2, axis=-1)

        rewards = jnp.squeeze(samples['reward'], axis=-1)
        old_Q1_errors = (rewards + V_tp1 - old_Q1)
        old_Q1_loss = optax.huber_loss(old_Q1_errors, delta=1.)  # TODO: Modify delta
        old_Q2_errors = (rewards + V_tp1 - old_Q2)
        old_Q2_loss = optax.huber_loss(old_Q2_errors, delta=1.)  # TODO: Modify delta

        critic_loss = old_Q1_loss + old_Q2_loss
        loss = jnp.mean(critic_loss)
        logs = {'critic_loss': critic_loss}
        return (loss, logs)

    def loss(self, online_params, target_params, state, samples):
        # For compatibility with BaseAlgorithm
        actor_loss, logs = self.actor_loss(
            online_params.actor, online_params.critic, state, samples)
        critic_loss, critic_logs = self.critic_loss(
            online_params.critic, online_params.actor, target_params, state, samples)
        loss = actor_loss + critic_loss
        
        # Update the logs
        logs.update(critic_logs)
        logs['loss'] = loss
        return (loss, logs)

    def init(self, key, samples, normalization=1):
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)

        # Initialize the network parameters (both online, and possibly target)
        actor_params, actor_state = self.actor_network.init(subkey1, samples['observation'])
        critic1_params, critic1_state = self.critic_network.init(subkey2, samples['observation'])
        critic2_params, critic2_state = self.critic_network.init(subkey3, samples['observation'])
        online_params = SACParameters(actor=actor_params, critic=(critic1_params, critic2_params))

        target_params = online_params if self.use_target else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Set the normalization to the size of the dataset
        actor_state['~']['normalization'] = jnp.full_like(
            actor_state['~']['normalization'], normalization)
        net_state = SACParameters(actor=actor_state, critic=(critic1_state, critic2_state))

        # Initialize the state of the optimizers
        opt_state = SACParameters(
            actor=self.optimizer.actor.init(online_params.actor),
            critic=self.optimizer.critic.init(online_params.critic),
        )

        # Initialize the state
        state = AlgoState(
            optimizer=opt_state,
            steps=jnp.array(0),
            network=net_state
        )

        return (params, state)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.actor_network.apply(params.actor, state.actor, observations)
        return log_pi

    def periodic_update_td3(self, params, params_critic, state, samples, logs, num_updates=1):
        params_actor = params.online.actor
        for _ in range(num_updates):
            (actor_loss, logs_actor), grads_actor = jax.value_and_grad(self.actor_loss, has_aux=True)(
                params_actor, params_critic, state.network, samples)  # Use the updated critic parameters

            updates_actor, opt_state_actor = self.optimizer.actor.update(
                grads_actor, state.optimizer.actor, params.online.actor)
            params_actor = optax.apply_updates(params.online.actor, updates_actor)

        logs.update(logs_actor)
        logs['loss'] += actor_loss
        return params_actor, opt_state_actor, logs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, state, samples):
        # Update of the parameters of the critic
        (critic_loss, logs), grads_critic = jax.value_and_grad(self.critic_loss, has_aux=True)(
            params.online.critic, params.online.actor, params.target, state.network, samples)

        updates_critic, opt_state_critic = self.optimizer.critic.update(
            grads_critic, state.optimizer.critic, params.online.critic)
        params_critic = optax.apply_updates(params.online.critic, updates_critic)

        logs['loss'] = critic_loss

        # Update the parameters of the actor with TD3 support
        params_actor, opt_state_actor, logs = optax.periodic_update(
            self.periodic_update_td3(params, params_critic, state, samples, logs, num_updates=self.policy_frequency),
            (params.online.actor, state.optimizer.actor, logs),
            state.steps + 1,
            self.policy_frequency
        )

        # Pack the parameters & optimizer state
        online_params = SACParameters(actor=params_actor, critic=params_critic)
        opt_state = SACParameters(actor=opt_state_actor, critic=opt_state_critic)

        # Update the target parameters
        if self.target == 'periodic':
            target_params = optax.periodic_update(
                jax.tree_util.tree_map(
                    lambda new, old: 0.005 * new + (1.0 - 0.005) * old,
                    online_params, params.target),
                params.target,
                state.steps + 1,
                **self.target_kwargs
            )

        elif self.target == 'incremental':
            target_params = optax.incremental_update(
                online_params,
                params.target,
                **self.target_kwargs
            )
        elif self.target is None:
            target_params = params.target
        else:
            raise ValueError(f'Unknown target: {self.target}')

        params = AlgoParameters(online=online_params, target=target_params)
        state = AlgoState(optimizer=opt_state, steps=state.steps + 1, network=state.network)

        # Update the logs
        # logs.update(logs_actor)
        # logs['loss'] = actor_loss + critic_loss
        return (params, state, logs)
