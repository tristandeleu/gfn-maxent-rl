import numpy as np


def evaluation(params_online, state_network, key, algorithm, env_valid, config):
    all_returns = []
    adjacencies = []
    returns = np.zeros((8,))
    observations_valid, _ = env_valid.reset()
    # Evaluation samples
    while len(all_returns) < config.evaluation_batch_size:
        # Sample actions from the model (w/o exploration)
        actions_valid, key, _ = algorithm.act(
            params_online, state_network, key, observations_valid, epsilon=1.)
        actions_valid = np.asarray(actions_valid)

        # Apply the actions in the evaluation environment
        observations_valid, rewards_valid, dones_valid, _, _ = env_valid.step(actions_valid)

        # Compute the return for the trajectories that are done
        returns = returns + rewards_valid * (1 - dones_valid)
        all_returns.extend([returns[i] for i, done_valid in enumerate(dones_valid) if done_valid])
        returns[dones_valid] = 0.

        # Adjacencies for the trajectories that are done
        adjacencies.extend([observations_valid['adjacency'][i] for i, done_valid in enumerate(dones_valid) if done_valid])

    # average_return = np.array(all_returns).mean()

    return np.array(all_returns).mean(), np.array(adjacencies)