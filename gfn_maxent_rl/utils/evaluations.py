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
    """Get samples by running the policy in the environment.

    Parameters
    ----------
    env : gym.vector.VectorEnv instance
        The environment.

    algorithm : BaseAlgorithm instance
        The algorithm. This must implement the `log_policy` method.

    params : Any
        The parameters of the networks (e.g., the policy network).
        Note that this must be the parameters of the *online* network
        (i.e., the parameters learned by the algorithm).

    net_state : Any
        The state of the network. This will be typically `state.network`,
        where `state` is the state returned by the initialization of the
        algorithm (and updated during training).

    key : jax.random.PRNGKey
        The Jax random key.

    num_samples : int
        The number of samples to return.

    copy_env : bool
        Whether `env` must be (deep)copied or not. This is useful if `env`
        is the instance of the environment used for training, to avoid any
        interaction. Set to False if we have an explicit validation env.

    verbose : bool
        Display a progress bar.

    Returns
    -------
    samples : list of samples
        A list of samples, of length `num_samples`. The samples are hashable
        keys, dependent on the environment (e.g., a tuple for Treesample envs).

    returns : np.ndarray, shape `(num_samples,)`
        The return of the trajectory for each sample. If the environment is not
        wrapped with a reward correction, then this corresponds exactly to
        the log-reward of each sample.
    """
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
