import jax.numpy as jnp
import jax
import haiku as hk
import optax

from collections import namedtuple
from abc import ABC, abstractmethod
from functools import partial

from gfn_maxent_rl.envs.dag_gfn.policy import uniform_log_policy


AlgoParameters = namedtuple('AlgoParameters', ['online', 'target'])
AlgoState = namedtuple('AlgoState', ['optimizer', 'steps', 'network'])


class BaseAlgorithm(ABC):
    def __init__(self, network, update_target_every=0):
        self.network = hk.without_apply_rng(hk.transform_with_state(network))
        self.update_target_every = update_target_every

        self._optimizer = None

    @abstractmethod
    def loss(self, online_params, target_params, state, samples):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, state, key, observations, epsilon):
        batch_size = observations['mask'].shape[0]
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Get the policies
        log_pi = self.log_policy(params, state, observations)  # Get the current policy
        log_uniform = uniform_log_policy(observations['mask'])  # Get uniform policy (exploration)

        # Mixture of the policies
        is_exploration = jax.random.bernoulli(subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_probs = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = jax.random.categorical(subkey2, logits=log_probs)

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
            'log_probs': log_probs,
        }
        return (actions, key, logs)

    def log_policy(self, params, state, observations):
        log_pi, _ = self.network.apply(params, state, observations['graph'], observations['mask'])
        return log_pi

    def step(self, params, state, samples):
        grads, logs = jax.grad(self.loss, has_aux=True)(params.online, params.target, state.network, samples)

        # Update the online parameters
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params.online)
        online_params = optax.apply_updates(params.online, updates)

        # Update the target parameters
        if self.update_target_every > 0:
            target_params = optax.periodic_update(
                online_params,
                params.target,
                state.steps + 1,
                self.update_target_every
            )
        else:
            target_params = params.target

        params = AlgoParameters(online=online_params, target=target_params)
        state = AlgoState(optimizer=opt_state, steps=state.steps + 1, network=state.network)

        return (params, state, logs)

    def init(self, key, samples, normalization=1.):
        # Initialize the network parameters (both online, and possibly target)
        online_params, net_state = self.network.init(key, samples['graph'], samples['mask'])
        target_params = online_params if (self.update_target_every > 0) else None
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

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                'model, you must set `model.optimizer = optax.sgd(...)` first.')
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = optax.chain(value, optax.zero_nans())
