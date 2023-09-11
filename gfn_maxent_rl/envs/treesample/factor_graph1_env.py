import numpy as np
import gym
import networkx as nx
import math

from gym.spaces import Box, Discrete
from numpy.random import default_rng

from gfn_maxent_rl.envs.dag_gfn.data_generation.graphs import sample_erdos_renyi_graph


class FactorGraph1Environment(gym.vector.VectorEnv):
    def __init__(self, num_envs, graph, potentials, permutation):
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
        self._index = np.zeros((num_envs,), dtype=np.int_)
        self._arange = np.arange(num_envs)

        observations_space = Box(
            low=-1,
            high=self.num_categories,
            shape=(self.num_variables,),
            dtype=np.int_
        )
        action_space = Discrete(self.num_categories)
        super().__init__(num_envs, observations_space, action_space)

    def reset(self, *, seed=None, options=None):
        self._state[:] = -1
        self._index[:] = 0
        return (np.copy(self._state), {})

    def step(self, actions):
        indices = self.permutation[self._index]
        self._state[self._arange, indices] = actions

        # Compute the rewards (more precisely, difference in log-rewards)
        rewards = np.zeros((self.num_envs,), dtype=np.float_)

        for clique, potential in self.potentials:
            # Check if the clique is active
            is_in_clique = np.any(clique == actions[:, None], axis=1)
            assignments = self._state[:, clique]
            full_assignment = np.all(assignments != -1, axis=1)
            is_active = np.logical_and(is_in_clique, full_assignment)

            # Get the codes for the assignments
            base = self.num_categories ** np.arange(len(clique))
            codes = np.sum(assignments[is_active] * base, axis=1)

            # Add the new potential
            rewards[is_active] += potential[codes]

        self._index = (self._index + 1) % self.num_variables
        dones = (self._index == 0)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)

        return (np.copy(self._state), rewards, dones, truncated, {})

    @classmethod
    def random(
        cls,
        num_envs,
        num_variables,
        num_categories,
        rng=default_rng(),
        max_retries=1000,
        max_clique_size=4,
    ):
        # Create a random graph (Erdos-Renyi) with a single connected component
        for _ in range(max_retries):
            graph = sample_erdos_renyi_graph(
                num_variables=num_variables,
                p=2 * math.log(num_variables) / num_variables,
                nodes=np.arange(num_variables),
                create_using=nx.Graph,
                rng=rng
            )
            if nx.number_connected_components(graph) == 1:
                # Ensure that there is no clique of size > max_clique_size
                cliques = list(nx.find_cliques(graph))
                if all(len(clique) <= max_clique_size for clique in cliques):
                    break
        else:
            raise RuntimeError('Unable to create a random graph over '
                f'{num_variables} variables with a single connected component, '
                f'or with all cliques of size < {max_clique_size}.')

        # Randomly generate the factors (potentials samples from normal distribution)
        potentials = []
        for clique in cliques:
            num_states = num_categories ** len(clique)
            potentials.append((
                np.asarray(clique, dtype=np.int_),
                rng.normal(loc=0., scale=1., size=(num_states,))
            ))
        
        # Find the variable ordering based on the heuristic
        permutation = []
        for clique in sorted(cliques, key=len, reverse=True):
            for variable in clique:
                if variable not in permutation:
                    permutation.append(variable)
        assert len(permutation) == num_variables
        permutation = np.asarray(permutation, dtype=np.int_)

        return cls(num_envs, graph, potentials, permutation)
