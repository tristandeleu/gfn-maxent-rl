import jax
import optax

from numpy.random import default_rng

from gfn_maxent_rl.envs.dag_gfn.factories import get_dag_gfn_env
from gfn_maxent_rl.algos.detailed_balance import GFNDetailedBalance
from gfn_maxent_rl.envs.dag_gfn.policy import policy_network

from gfn_maxent_rl.utils.estimation import estimate_log_probs_backward, estimate_log_probs_beam_search

env, infos = get_dag_gfn_env(
    artifact='tristandeleu_mila_01/gfn_maxent_rl/er2-lingauss-d005:v0',
    prior_name='uniform',
    score_name='bge',
    seed=0
)

algorithm = GFNDetailedBalance(
    env=env,
    network=policy_network
)
algorithm.optimizer = optax.adam(1e-3)

key = jax.random.PRNGKey(0)
params, state = algorithm.init(key)

samples = [
    frozenset({(0, 1), (1, 2), (2, 4), (3, 2)}),
    frozenset({}),
    frozenset({(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}),
    frozenset({(3, 0), (1, 2), (0, 4), (1, 4)}),
]

log_probs = estimate_log_probs_backward(
    env,
    algorithm,
    params.online,
    state.network,
    samples=samples,
    rng=default_rng(0),
    batch_size=2,
    num_trajectories=1000,
    verbose=True
)

print('Backward estimation', log_probs)

samples = [
    frozenset({(0, 1), (1, 2), (2, 4), (3, 2)})
]

log_probs = estimate_log_probs_beam_search(
    env,
    algorithm,
    params.online,
    state.network,
    samples=samples,
    rng=default_rng(0),
    batch_size=2,
    beam_size=10,
    num_trajectories=10,
    verbose=True
)

print('Beam search estimation', log_probs)
