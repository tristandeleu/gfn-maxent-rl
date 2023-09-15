import numpy as np
import gym
import math

from gym.spaces import Dict, Box, Discrete

from gfn_maxent_rl.envs.dag_gfn.jraph_utils import to_graphs_tuple, batch_sequences_to_graphs_tuple
from gfn_maxent_rl.envs.dag_gfn.policy import uniform_log_policy


class DAGEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, joint_model):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.

        joint_model : JointModel instance
            The joint model that computes P(D, G) = P(D | G)P(G).
        """
        self.joint_model = joint_model
        self.num_variables = joint_model.num_variables
        self._state = None

        shape = (self.num_variables, self.num_variables)
        observation_space = Dict({
            'adjacency': Box(low=0., high=1., shape=shape, dtype=np.float32),
            'mask': Box(low=0., high=1., shape=shape, dtype=np.float32),
            # We do not include 'graph' in `observation_space` to avoid automatic batching
        })
        action_space = Discrete(self.num_variables ** 2 + 1)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, *, seed=None, options=None):
        shape = (self.num_envs, self.num_variables, self.num_variables)
        closure_T = np.eye(self.num_variables, dtype=np.bool_)
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.bool_),
            'closure_T': np.tile(closure_T, (self.num_envs, 1, 1)),
        }
        return (self.observations(), {})

    def step(self, actions):
        sources, targets = divmod(actions, self.num_variables)
        dones = (sources == self.num_variables)
        sources, targets = sources[~dones], targets[~dones]
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)

        # Make sure that all the actions are valid
        is_invalid = np.logical_or(self._state['adjacency'], self._state['closure_T'])
        if np.any(is_invalid[~dones, sources, targets]):
            raise ValueError('Some actions are invalid: either the edge to be '
                'added is already in the DAG, or adding this edge would lead to a cycle.')

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        rewards[~dones] = self.joint_model.delta_score(
            self._state['adjacency'][~dones], sources, targets)

        # Update the adjacency matrices
        self._state['adjacency'][~dones, sources, targets] = True
        self._state['adjacency'][dones] = False

        # Update the transpose of the transitive closures
        source_rows = np.expand_dims(self._state['closure_T'][~dones, sources, :], axis=1)
        target_cols = np.expand_dims(self._state['closure_T'][~dones, :, targets], axis=2)
        self._state['closure_T'][~dones] |= np.logical_and(source_rows, target_cols)  # Outer product
        self._state['closure_T'][dones] = np.eye(self.num_variables, dtype=np.bool_)

        return (self.observations(), rewards, dones, truncated, {})

    def observations(self):
        return {
            'adjacency': self._state['adjacency'].astype(np.float32),
            'mask': 1. - np.asarray(self._state['adjacency'] + self._state['closure_T'], dtype=np.float32),
            'graph': to_graphs_tuple(self._state['adjacency'])
        }

    # Properties & methods to interact with the replay buffer

    @property
    def observation_dtype(self):
        nbytes = math.ceil((self.num_variables ** 2) / 8)
        return np.dtype([
            ('adjacency', np.uint8, (nbytes,)),
            ('mask', np.uint8, (nbytes,)),
        ])

    @property
    def max_length(self):
        return (self.num_variables * (self.num_variables - 1) // 2) + 1

    def encode(self, observations):
        def _encode(decoded):
            encoded = decoded.reshape(-1, self.num_variables ** 2)
            return np.packbits(encoded.astype(np.int32), axis=1)

        batch_size = observations['adjacency'].shape[0]
        encoded = np.empty((batch_size,), dtype=self.observation_dtype)
        encoded['adjacency'] = _encode(observations['adjacency'])
        encoded['mask'] = _encode(observations['mask'])
        return encoded

    def _decode(self, encoded):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(np.float32)

    def decode(self, observations):
        adjacency = self._decode(observations['adjacency'])
        return {
            'adjacency': adjacency,
            'mask': self._decode(observations['mask']),
            'graph': to_graphs_tuple(adjacency)
        }

    def decode_sequence(self, samples):
        return {
            'adjacency': self._decode(samples['observations']['adjacency']),
            'mask': self._decode(samples['observations']['mask']),
            'graph': batch_sequences_to_graphs_tuple(
                self.num_variables, samples['actions'], samples['lengths'])
        }


    # Method to interact with the algorithm (uniform sampling of action)

    def uniform_log_policy(self, observations):
        return uniform_log_policy(observations['mask'])

    def num_parents(self, observations):
        return observations['graph'].n_edge[:-1]  # [:-1] -> Remove padding
