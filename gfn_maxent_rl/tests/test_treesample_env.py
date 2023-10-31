from numpy.random import default_rng

from gfn_maxent_rl.envs.treesample.chain_env import chain_env

rng = default_rng(0)
env = chain_env(
    num_envs=8,
    num_variables=3,
    num_categories=2,
    rng=rng,
)

# Ensure the MDP is well formed
mdp = env.mdp_state_graph

# The "+ 1" is due to the "unassigned value"
assert len(mdp) == (env.num_categories + 1) ** env.num_variables

# All edges corresponds to assigning exactly 1 unassigned value
for edge in mdp.edges:
    num_diffs = tuple(x for (x, y) in zip(*edge) if x != y)
    assert num_diffs == (-1,)

for keys, observations in env.all_states_batch_iterator(batch_size=3):
    rewards = env.log_reward(observations)
    print(keys, rewards)
