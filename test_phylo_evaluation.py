import numpy as np
import jax.numpy as jnp
import jax
import optax
import networkx as nx

from numpy.random import default_rng

from gfn_maxent_rl.data.replay_buffer import ReplayBuffer
from gfn_maxent_rl.envs.phylo_gfn.factories import get_phylo_gfn_env
from gfn_maxent_rl.algos.detailed_balance_vanilla import GFNDetailedBalanceVanilla
from gfn_maxent_rl.envs.phylo_gfn.policy import policy_network_transformer
from gfn_maxent_rl.envs.phylo_gfn.policy import f_network_transformer
from gfn_maxent_rl.algos.detailed_balance_vanilla import DBVParameters

from gfn_maxent_rl.utils.estimation import estimate_log_probs_backward, estimate_log_probs_beam_search
from gfn_maxent_rl.utils.evaluations import get_samples_from_env

from pathlib import Path
from numpy.random import default_rng

from gfn_maxent_rl.utils import io
import os
import wandb
import sys
sys.path.insert(0, '..')

# ---------- Load the model ------------
# # load a model
# params.online = io.load(os.path.join('/home/mila/p/padideh.nouri', 'model.npz'))
# my_model = wandb.restore('model.npz', run_path='tristandeleu_mila_01/gfn_maxent_rl/runs/aw4ru2l7')

api = wandb.Api()
run = api.run('tristandeleu_mila_01/gfn_maxent_rl/aw4ru2l7')

# Check the algorithm (we need to pack the parameters)
# run.config['exp_name_algorithm']

root = Path(os.getenv('SLURM_TMPDIR')) / run.id
run.file('model.npz').download(root=root, exist_ok=True)

with open(root / 'model.npz', 'rb') as f:
    params_online = DBVParameters(**io.load(f))

# ---------- Load the environment ------------
DATA_FOLDER = Path('./gfn_maxent_rl/envs/phylo_gfn/datasets')
env, infos = get_phylo_gfn_env(
    dataset_name="DS1",
    num_envs=1,
    data_folder=DATA_FOLDER
)

replay = ReplayBuffer(100, env)

algorithm = GFNDetailedBalanceVanilla(
    env=env,
    policy_network=policy_network_transformer,
    flow_network=f_network_transformer,
)
algorithm.optimizer = DBVParameters(policy=optax.adam(1e-3), flow=optax.adam(1e-3))

key = jax.random.PRNGKey(0)
params, state = algorithm.init(key)


# ---------- Test the estimation ------------
samples = get_samples_from_env(env=env,
                               algorithm=algorithm,
                               params=params_online,
                               net_state=state.network,
                               num_samples=100,
                               key=key,
                               )

log_probs = estimate_log_probs_backward(
    env,
    algorithm,
    params_online,
    state.network,
    samples=samples[0],
    rng=default_rng(0),
    batch_size=2,
    num_trajectories=100,
    verbose=True
)

print('Backward estimation', log_probs)

# samples = [
#     frozenset({(0, 1), (1, 2), (2, 4), (3, 2)})
# ]
# samples = get_samples_from_env(env=env,
#                                algorithm=algorithm,
#                                params=params.online,
#                                net_state=state.network,
#                                num_samples=100,
#                                key=key,
#                                )
#
# log_probs = estimate_log_probs_beam_search( # only for dag-gfn env
#     env,
#     algorithm,
#     params.online,
#     state.network,
#     samples=samples[0],
#     rng=default_rng(0),
#     batch_size=2,
#     beam_size=10,
#     num_trajectories=10,
#     verbose=True
# )
#
# print('Beam search estimation', log_probs)
