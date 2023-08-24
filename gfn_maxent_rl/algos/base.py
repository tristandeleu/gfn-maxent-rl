import jax.numpy as jnp
import jax
import haiku as hk
import optax

from collections import namedtuple
from abc import ABC, abstractmethod


AlgoParameters = namedtuple('AlgoParameters', ['online', 'target'])
AlgoState = namedtuple('AlgoState', ['optimizer', 'steps'])


class BaseAlgorithm(ABC):
    def __init__(self, network, update_target_every=0):
        self.network = hk.without_apply_rng(hk.transform(network))
        self.update_target_every = update_target_every

        self._optimizer = None

    @abstractmethod
    def loss(self, online_params, target_params, samples):
        pass

    def act(self, params, key, observations, epsilon):
        batch_size = observations.mask.shape[0]
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Get the policies
        log_pi = self.log_policy(params, observations)  # Get the current policy
        log_uniform = None  # TODO

        # Mixture of the policies
        is_exploration = jax.random.bernoulli(subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = None  # TODO

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32)
        }
        return (actions, key, logs)

    def log_policy(self, params, observations):
        return self.network.apply(params, observations.graph, observations.mask)

    def step(self, params, state, samples):
        grads, logs = jax.grad(self.loss, has_aux=True)(params.online, params.target, samples)

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
        state = AlgoState(optimizer=opt_state, steps=state.steps + 1)

        return (params, state, logs)

    def init(self, key, samples):
        # Initialize the network parameters (both online, and possibly target)
        online_params = self.network.init(key, samples['graph'], samples['mask'])
        target_params = online_params if (self.update_target_every > 0) else None
        params = AlgoParameters(online=online_params, target=target_params)

        # Initialize the state
        state = AlgoState(
            optimizer=self.optimizer.init(online_params),
            steps=jnp.array(0)
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
