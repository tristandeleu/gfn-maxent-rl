import numpy as np
import gym
import math
import networkx as nx

from gym.spaces import Dict, Box, Discrete, MultiBinary
from itertools import product, chain
from numpy.random import default_rng
from scipy.special import gammaln

from gfn_maxent_rl.envs.treesample.policy import uniform_log_policy, action_mask
from gfn_maxent_rl.envs.errors import StatesEnumerationError
from gfn_maxent_rl.envs.treesample.functional import reset, step, state_to_observation


class FactorGraphEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, graph, potentials, permutation):
        """Environment for inference over a factor graph defined by its potentials.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments, or equivalently the number of
            parallel trajectories to sample.

        graph : nx.Graph instance
            The structure of the factor graph (as a Markov network).

        potentials : dict
            A dictionary of (clique, potential), where `clique` is a
            representation of a clique (an array of node indices), and
            potential is an array containing the potentials of all
            possible assignments of the variables in the clique (array
            of size num_categories ** (number of variables in clique)).

        permutation : np.ndarray
            An array of indices for a fixed order in which the variables
            can be assigned (found by heuristics, unused in general).
        """
        self.graph = graph
        self.potentials = potentials
        self.permutation = permutation
        self.num_variables = len(permutation)
        
        # Get the number of categories
        clique, potential = potentials[0]
        self.num_categories = round(math.exp(math.log(len(potential)) / len(clique)))
        for clique, potential in potentials:
            assert potential.size == self.num_categories ** len(clique)

        self._state = np.full((num_envs, self.num_variables), -1, dtype=np.int_)
        self._all_states, self._all_keys = None, None
        self._state_graph = None

        observation_space = Dict({
            'variables': Box(
                low=-1,
                high=self.num_categories,
                shape=(self.num_variables,),
                dtype=np.int_
            ),
            'mask': MultiBinary(self.num_variables),
        })
        action_space = Discrete(self.num_variables * self.num_categories + 1)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, *, seed=None, options=None):
        self._state[:] = -1
        return (self.observations(), {})

    def step(self, actions):
        indices, values = divmod(actions, self.num_categories)
        dones = (indices == self.num_variables)

        if np.any(self._state[dones] == -1):
            raise RuntimeError('Invalid action: calling the stop action even '
                'though some variables have not been assigned.')
        
        indices, values = indices[~dones], values[~dones]

        if np.any(self._state[~dones, indices] != -1):
            raise RuntimeError('Invalid action: trying to set a variable '
                'that has already been assigned.')

        self._state[~dones, indices] = values

        # Compute the rewards (more precisely, difference in log-rewards)
        rewards = np.zeros((self.num_envs,), dtype=np.float_)
        active_potentials = np.zeros((self.num_envs, len(self.potentials)), dtype=np.bool_)

        if len(indices) > 0:
            for i, (clique, potential) in enumerate(self.potentials):
                # Check if the clique is active
                is_in_clique = np.any(clique == indices[:, None], axis=1)
                assignments = self._state[:, clique][~dones]
                full_assignment = np.all(assignments != -1, axis=1)
                is_active = np.logical_and(is_in_clique, full_assignment)

                # Get the codes for the assignments
                base = self.num_categories ** np.arange(len(clique))
                codes = np.sum(assignments[is_active] * base, axis=1)

                # Add the new potential
                rewards[is_active] += potential[codes]
                active_potentials[:, i] = is_active

        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        self._state[dones] = -1  # Clear state for complete trajectories
        rewards[dones] = 0.  # Terminal action has 0 reward
        infos = {'active_potentials': active_potentials}

        return (self.observations(), rewards, dones, truncated, infos)

    def observations(self):
        return {
            'variables': np.copy(self._state),
            'mask': (self._state == -1).astype(np.int_)
        }

    # Properties & methods to interact with the replay buffer

    @property
    def observation_dtype(self):
        return np.dtype([
            ('variables', np.int_, (self.num_variables,)),
            ('mask', np.int_, (self.num_variables,))
        ])

    @property
    def max_length(self):
        return self.num_variables + 1

    def encode(self, observations):
        batch_size = observations['variables'].shape[0]
        encoded = np.empty((batch_size,), dtype=self.observation_dtype)
        encoded['variables'] = observations['variables']
        encoded['mask'] = observations['mask']
        return encoded

    def decode(self, observations):
        return {
            'variables': observations['variables'].astype(np.int32),
            'mask': observations['mask'].astype(np.float32),
        }

    @property
    def observation_sequence_dtype(self):
        return self.observation_dtype

    def encode_sequence(self, observations):
        return self.encode(observations)

    def decode_sequence(self, samples):
        return self.decode(samples['observations'])

    # Method to interact with the algorithm (uniform sampling of action)

    def uniform_log_policy(self, observations):
        return uniform_log_policy(observations['mask'], self.num_categories)

    def num_parents(self, observations):
        return (observations['variables'] != -1).sum(axis=-1)

    def action_mask(self, observations):
        return action_mask(observations['mask'], self.num_categories)

    # Method for evaluation

    def all_states_batch_iterator(self, batch_size, terminating=False):
        num_states = (self.num_categories + 1) ** self.num_variables

        if num_states > 1e5:
            raise StatesEnumerationError('Impossible to enumerate all the '
                f'states for `num_categories = {self.num_categories}` & '
                f'`num_variables = {self.num_variables} (the number of states '
                f'= {num_states} > 100k).')

        if self._all_states is None:
            iterator = product(range(-1, self.num_categories), repeat=self.num_variables)
            self._all_keys = list(iterator)
            self._all_states = np.fromiter(
                chain(*self._all_keys),
                dtype=np.int_,
                count=num_states * self.num_variables,
            ).reshape(-1, self.num_variables)
        
        if terminating:
            is_terminating_state = np.all(self._all_states != -1, axis=1)
            states = self._all_states[is_terminating_state]
            keys = [key for (key, is_terminating)
                in zip(self._all_keys, is_terminating_state) if is_terminating]
        else:
            states, keys = self._all_states, self._all_keys
        
        for index in range(0, states.shape[0], batch_size):
            slice_ = slice(index, index + batch_size)
            variables = states[slice_]
            observations = {
                'variables': variables.astype(np.int32),
                'mask': (variables == -1).astype(np.float32),
            }
            yield (keys[slice_], observations)

    def log_reward(self, observations):
        variables = observations['variables']
        log_rewards = np.zeros((variables.shape[0],), dtype=np.float_)

        for clique, potential in self.potentials:
            assignments = variables[:, clique]

            # Get the codes for the assignments
            base = self.num_categories ** np.arange(len(clique))
            codes = np.sum(assignments * base, axis=1)

            # Add the new potential
            log_rewards += potential[codes]

        return log_rewards

    @property
    def mdp_state_graph(self):
        num_states = (self.num_categories + 1) ** self.num_variables

        if num_states > 1e5:
            raise StatesEnumerationError('Impossible to enumerate all the '
                f'states for `num_categories = {self.num_categories}` & '
                f'`num_variables = {self.num_variables} (the number of states '
                f'= {num_states} > 100k).')

        if self._state_graph is None:
            states = list(product(range(-1, self.num_categories), repeat=self.num_variables))
            terminating_states = product(range(self.num_categories), repeat=self.num_variables)

            edges = []
            for state in states:
                for i, variable in enumerate(state):
                    if variable == -1:
                        edges.extend([
                            (state, state[:i] + (value,) + state[i+1:],
                            {'action': i * self.num_categories + value})
                            for value in range(self.num_categories)
                        ])

            # Create the MDP (graph over states)
            self._state_graph = nx.DiGraph(initial=(-1,) * self.num_variables)
            self._state_graph.add_nodes_from(states, terminating=False)
            nx.set_node_attributes(
                self._state_graph,
                {state: {'terminating': True} for state in terminating_states}
            )
            self._state_graph.add_edges_from(edges)

        return self._state_graph

    def observation_to_key(self, observations):
        return [tuple(variables) for variables in observations['variables']]

    def key_batch_iterator(self, keys, batch_size):
        for index in range(0, len(keys), batch_size):
            yield (keys[index:index + batch_size], self.max_length)

    def key_to_action_mask(self, keys):
        action_masks = np.zeros((len(keys), self.single_action_space.n), dtype=np.bool_)
        for i, variables in enumerate(keys):
            indices = np.array([i * self.num_categories + value
                for (i, value) in enumerate(variables)])
            action_masks[i, indices] = True
        return action_masks

    def backward_sample_trajectories(
            self,
            keys,
            num_trajectories,
            max_length=None,
            blacklist=None,
            rng=default_rng(),
            max_retries=10,
    ):
        if blacklist is None:
            blacklist = dict((key, set()) for key in keys)

        trajectories = np.full((len(keys), num_trajectories, self.max_length), -1, dtype=np.int_)

        for i, key in enumerate(keys):
            actions = np.array([i * self.num_categories + value
                for (i, value) in enumerate(key)])
            actions = np.repeat(actions[None], num_trajectories, axis=0)

            idx, offset = 0, 0
            while (offset < num_trajectories) and (idx < max_retries):
                new_trajs = np.full((num_trajectories, self.max_length),
                    self.single_action_space.n - 1, dtype=np.int_)
                new_trajs[:, :-1] = rng.permuted(actions, axis=1)

                # Get the indices of the whitelisted trajectories
                is_whitelist = np.array([tuple(traj) not in blacklist[key]
                    for traj in new_trajs], dtype=np.bool_)
                num_whitelist = np.sum(is_whitelist)

                trajectories[i, offset:offset + num_whitelist] = new_trajs[is_whitelist]
                offset += num_whitelist
                idx += 1

            if idx == max_retries:
                raise RuntimeError('Impossible to find non-blacklisted trajectories')

        # Log-backward probabilities
        log_pB = np.full((len(keys), num_trajectories), -gammaln(self.num_variables + 1))

        return (trajectories, log_pB)

    # Functional API

    def func_reset(self, batch_size):
        return reset(batch_size, self.num_variables)

    def func_step(self, states, actions):
        return step(states, actions, self.num_categories)
    
    def func_state_to_observation(self, states, trajectories):
        return state_to_observation(states)
