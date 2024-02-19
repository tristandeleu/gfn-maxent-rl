import numpy as np
import jax.numpy as jnp
import jax
import haiku as hk
import optax

from collections import namedtuple
from abc import ABC, abstractmethod
from functools import partial


AlgoParameters = namedtuple('AlgoParameters', ['online', 'target'])
AlgoState = namedtuple('AlgoState', ['optimizer', 'steps', 'network'])


class BaseAlgorithm(ABC):
    def __init__(self, env, target=None, target_kwargs={}):
        self._optimizer = None
        self.env = env
        self.target = target
        self.target_kwargs = target_kwargs

    @abstractmethod
    def loss(self, online_params, target_params, state, samples):
        pass

    @abstractmethod
    def log_policy(self, params, state, observations):
        pass

    @abstractmethod
    def init(self, key, samples, **kwargs):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def act(self, params, state, key, observations, epsilon):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Get the policies
        log_pi = self.log_policy(params, state, observations)  # Get the current policy
        log_uniform = self.env.uniform_log_policy(observations)  # Get uniform policy (exploration)

        # Mixture of the policies
        batch_size = log_pi.shape[0]
        is_exploration = jax.random.bernoulli(subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_probs = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = jax.random.categorical(subkey2, logits=log_probs)

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
            'log_probs': log_probs,
        }
        return (actions, key, logs)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, state, samples):
        grads, logs = jax.grad(self.loss, has_aux=True)(params.online, params.target, state.network, samples)

        # Update the online parameters
        updates, opt_state = self.optimizer.update(grads, state.optimizer, params.online)
        online_params = optax.apply_updates(params.online, updates)

        # Update the target parameters
        if self.target == 'periodic':
            target_params = optax.periodic_update(
                online_params,
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

        return (params, state, logs)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                'model, you must set `model.optimizer = optax.sgd(...)` first.')
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        if not isinstance(value, optax.GradientTransformation):
            fields = value._fields
            value = optax.multi_transform(
                value._asdict(),
                type(value)(**dict(zip(fields, fields)))
            )

        self._optimizer = optax.chain(value, optax.zero_nans())

    @property
    def use_target(self):
        return self.target is not None

    @property
    def _dummy_observation(self):
        observation = np.zeros((1,), dtype=self.env.observation_dtype)
        return self.env.decode(observation)


GFNParameters = namedtuple('GFNParameters', ['network', 'log_Z'])

class GFNBaseAlgorithm(BaseAlgorithm):
    def __init__(self, env, network, target=None, target_kwargs={}):
        super().__init__(env, target=target, target_kwargs=target_kwargs)
        self.network = hk.without_apply_rng(hk.transform_with_state(network))

    def init(self, key, normalization=1):
        # Initialize the network parameters (both online, and possibly target)
        net_params, net_state = self.network.init(key, self._dummy_observation)
        online_params = GFNParameters(network=net_params, log_Z=jnp.array(0.))
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
        log_pi, _ = self.network.apply(params.network, state, observations)
        return log_pi
