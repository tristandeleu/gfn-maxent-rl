import numpy as np

from copy import deepcopy
from tqdm.auto import trange


def evaluation(params_online, state_network, key, algorithm, env_valid, config):
    all_returns = []
    adjacencies = []
    returns = np.zeros((8,))
    observations, _ = env_valid.reset()
    # observations_prev = np.copy(observations)
    # Evaluation samples
    while len(all_returns) < config.evaluation_batch_size:
        # Sample actions from the model (w/o exploration)
        actions_valid, key, _ = algorithm.act(
            params_online, state_network, key, observations, epsilon=1.)
        actions_valid = np.asarray(actions_valid)

        # Apply the actions in the evaluation environment
        next_observations, rewards_valid, dones_valid, _, _ = env_valid.step(actions_valid)


        # Compute the return for the trajectories that are done
        returns = returns + rewards_valid * (1 - dones_valid)
        all_returns.extend([returns[i] for i, done_valid in enumerate(dones_valid) if done_valid])
        returns[dones_valid] = 0.

        # Adjacencies for the trajectories that are done
        adjacencies.extend([observations['adjacency'][i] for i, done_valid in enumerate(dones_valid) if done_valid])
        observations = next_observations

    # average_return = np.array(all_returns).mean()

    return np.array(all_returns).mean(), np.array(adjacencies)


def get_samples_from_env(
        env,
        algorithm,
        params,
        net_state,
        key,
        num_samples=1000,
        copy_env=True,
        verbose=False,
        **kwargs
    ):
    samples, returns = [], []
    if copy_env:
        env = deepcopy(env)
    observations, _ = env.reset()

    returns_ = np.zeros((env.num_envs,), dtype=np.float_)
    with trange(num_samples, disable=(not verbose), **kwargs) as pbar:
        while len(samples) < num_samples:
            keys = env.observation_to_key(observations)

            # Sample actions from the model (w/o exploration)
            actions, key, _ = algorithm.act(
                params, net_state, key, observations, epsilon=1.)
            actions = np.asarray(actions)

            # Apply the actions in the environment
            observations, rewards, dones, *_ = env.step(actions)

            # Compute the returns
            returns_ = returns_ + rewards * (1. - dones)

            # Add samples from the complete trajectories
            samples.extend([key for (key, done) in zip(keys, dones) if done])
            returns.extend([return_ for (return_, done) in zip(returns_, dones) if done])
            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))

            # Reset the returns for complete trajectories
            returns_[dones] = 0.

    samples = samples[:num_samples]
    returns = returns[:num_samples]

    return (samples, np.asarray(returns))
