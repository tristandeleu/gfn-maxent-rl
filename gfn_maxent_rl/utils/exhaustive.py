import numpy as np

from scipy.special import logsumexp


def exact_log_posterior(env, batch_size=256):
    log_posterior = dict()

    for keys, observations in env.all_states_batch_iterator(batch_size):
        log_rewards = env.log_reward(observations)
        log_posterior.update(zip(keys, log_rewards))

    # Compute the log-partition function
    log_rewards = np.asarray(log_posterior.values())
    log_Z = logsumexp(log_rewards)

    # Normalize the rewards
    for key, log_reward in log_posterior.items():
        log_posterior[key] = log_reward - log_Z

    return log_posterior
