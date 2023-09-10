import numpy as np

from numpy.random import default_rng
from tqdm.auto import trange

from gfn_maxent_rl.envs.dag_gfn import DAGEnvironment, RewardCorrection
from gfn_maxent_rl.envs.dag_gfn.graph_priors import UniformPrior
from gfn_maxent_rl.envs.dag_gfn.scores import ZeroScore
from gfn_maxent_rl.envs.dag_gfn.base import JointModel
from gfn_maxent_rl.data.replay_buffer_episodic import EpisodicReplayBuffer

num_variables = 3

def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)
    logp_stop = -np.log1p(np.sum(masks, axis=1, keepdims=True))
    logp_continue = np.where(masks, logp_stop, -np.inf)
    return np.concatenate((logp_continue, logp_stop), axis=1)

def sample_random_actions(masks, rng):
    log_probs = uniform_log_policy(masks)
    z = rng.gumbel(size=log_probs.shape)
    return np.argmax(log_probs + z, axis=1)

joint_model = JointModel(
    graph_prior=UniformPrior(num_variables=num_variables),
    marginal_likelihood=ZeroScore(data=np.zeros((100, num_variables)))
)

rng = default_rng(0)

env = DAGEnvironment(num_envs=8, joint_model=joint_model)
observations, _ = env.reset()

replay = EpisodicReplayBuffer(capacity=10, num_variables=num_variables)
indices = None

for _ in range(100):
    # Sample random actions
    actions = sample_random_actions(observations['mask'], rng)

    # Step in the environment
    next_observations, rewards, dones, _, _ = env.step(actions)

    # Add to the replay buffer
    indices = replay.add(observations, actions, rewards, dones, next_observations, indices=indices)

    observations = next_observations
