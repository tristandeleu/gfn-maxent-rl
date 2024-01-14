import numpy as np
import jax.numpy as jnp
import jax
import optax
import networkx as nx

from gfn_maxent_rl.data.replay_buffer import ReplayBuffer
from gfn_maxent_rl.envs.dag_gfn.factories import get_dag_gfn_env
from gfn_maxent_rl.algos.detailed_balance import GFNDetailedBalance
from gfn_maxent_rl.envs.dag_gfn.policy import policy_network

from gfn_maxent_rl.utils.beam_search import beam_search_forward

env, infos = get_dag_gfn_env(
    artifact='tristandeleu_mila_01/gfn_maxent_rl/er2-lingauss-d005:v0',
    prior_name='uniform',
    score_name='bge',
    seed=0
)

replay = ReplayBuffer(100, env)

algorithm = GFNDetailedBalance(
    env=env,
    network=policy_network
)
algorithm.optimizer = optax.adam(1e-3)

key = jax.random.PRNGKey(0)
params, state = algorithm.init(key, replay.dummy_samples)

beam_search = beam_search_forward(env, algorithm, beam_size=10)
beam_search = jax.jit(beam_search)

# Get action masks
# graph = nx.to_numpy_array(infos['graph'], weight=None)
graph = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
])
sources, targets = np.nonzero(graph)
action_mask = np.zeros((env.single_action_space.n,), dtype=np.bool_)
action_mask[sources * env.num_variables + targets] = True

trajectories, log_probs, logs = beam_search(params.online, state.network, action_mask)

print(np.nonzero(action_mask))
print(trajectories)
print(log_probs)
print(logs)