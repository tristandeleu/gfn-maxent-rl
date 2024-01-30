import numpy as np
import jax.numpy as jnp
import jax
import math

from numpy.random import default_rng
from scipy.special import logsumexp
from tqdm.auto import tqdm

from gfn_maxent_rl.utils.beam_search import beam_search_forward


def estimate_log_probs_beam_search(
        env,
        algorithm,
        params,
        net_state,
        samples,
        rng=default_rng(),
        beam_size=128,
        batch_size=1,
        num_trajectories=1000,
        verbose=False,
        **kwargs
):
    log_probs = dict()

    # Vmap function over multiple samples
    log_prob_fn = log_prob_trajectories(env, algorithm)
    log_prob_fn = jax.vmap(log_prob_fn, in_axes=(None, None, 0))
    log_prob_fn = jax.jit(log_prob_fn)

    num_batches = math.ceil(len(samples) / batch_size)
    for keys, max_length in tqdm(env.key_batch_iterator(samples, batch_size=batch_size),
            total=num_batches, disable=(not verbose), **kwargs):
        # Vmap beam search over multiple samples
        beam_search = beam_search_forward(env, algorithm, beam_size=beam_size, max_length=max_length)
        beam_search = jax.vmap(beam_search, in_axes=(None, None, 0))
        beam_search = jax.jit(beam_search)

        # Run beam search on the samples
        action_masks = env.key_to_action_mask(keys)
        fwd_trajectories, fwd_log_probs, logs = beam_search(params, net_state, action_masks)
        if not np.all(logs['is_valid_length']):
            raise RuntimeError('Some trajectories are longer than the maximum length.')
        if not np.all(logs['is_valid_trajectories']):
            raise RuntimeError('Not all the trajectories lead to the same state.')
        fwd_log_probs = logsumexp(fwd_log_probs, axis=1)

        # Complement with randomly sampled trajectories
        blacklist = dict((key, set(map(tuple, trajs.tolist()))) for (key, trajs) in zip(keys, fwd_trajectories))
        bwd_trajectories, log_pB_bwd_trajs = env.backward_sample_trajectories(
            keys, num_trajectories, max_length=max_length, blacklist=blacklist, rng=rng)
        
        # Compute the log-probabilities of the backward trajectories
        log_pF_bwd_trajs = log_prob_fn(params, net_state, bwd_trajectories)

        # Average over all trajectories, and offset by total number of trajectories
        # Assumption: the beam-size is smaller than the total number of trajectories
        # TODO: Correct for rejection sampling, based on log p_B of the trajectories in beam search
        bwd_log_probs = logsumexp(log_pF_bwd_trajs - log_pB_bwd_trajs, axis=1) - math.log(num_trajectories)

        # Store the log-probabilities
        log_probs_ = np.logaddexp(fwd_log_probs, bwd_log_probs)
        log_probs.update(zip(keys, log_probs_))

    return log_probs


def estimate_log_probs_backward(
        env,
        algorithm,
        params,
        net_state,
        samples,
        rng=default_rng(),
        batch_size=1,
        num_trajectories=1000,
        verbose=False,
        **kwargs
):
    """Estimation of the log-probability of samples.

    Given a policy \pi(s_t+1 | s_t), the log-probability of a sample "x"
    is equal to

        P(x) = \sum_{tau} \pi(tau)
    
    where the sum is over all the trajectories from the initial state
    to the state "x", and \pi(tau) = \prod \pi(s_t+1 | s_t) along the
    trajectory. Since the number of trajectories is combinatorially large,
    we write this sum as

        P(x) = E_{tau ~ P_B}[\pi(tau) / P_B(tau)]
    
    where the expectation is over trajectories sampled with the backward
    policy P_B, and we used importance sampling. We can then use Monte-Carlo
    estimation to estimate this expectation, by sampling multiple trajectories
    using (uniform) P_B.

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

    samples : list
        The list of samples. Each sample must be a hashable key. See
        `utils/evaluations.py:get_samples_from_env`.

    rng : numpy.random.Generator instance
        The RNG for numpy (to sample trajectories with P_B).

    batch_size : int
        The batch-size for calling the evaluation function. Increase or
        decrease depending on the amount of (GPU) memory available.

    num_trajectories : int
        The number of trajectories for the Monte-Carlo estimation of the
        expectation above.

    verbose : bool
        Display a progress bar.

    Returns
    -------
    log_probs : dict (sample, float)
        A dictionary containing the estimate of the log-probability for
        each sample (samples are in the keys of the dictionary).
    """
    log_probs = dict()

    # Vmap function over multiple samples
    log_prob_fn = log_prob_trajectories(env, algorithm)
    log_prob_fn = jax.vmap(log_prob_fn, in_axes=(None, None, 0))
    log_prob_fn = jax.jit(log_prob_fn)

    num_batches = math.ceil(len(samples) / batch_size)
    for keys, max_length in tqdm(env.key_batch_iterator(samples, batch_size=batch_size),
            total=num_batches, disable=(not verbose), **kwargs):
        # Sample random trajectories
        trajectories, log_pB = env.backward_sample_trajectories(
            keys, num_trajectories, max_length=max_length, rng=rng)

        # Compute the log-probabilities of all trajectories
        log_pF = log_prob_fn(params, net_state, trajectories)
        log_pF = np.asarray(log_pF)

        # Average over trajectories (in log-space), with importance sampling
        log_probs_ = logsumexp(log_pF - log_pB, axis=1) - math.log(num_trajectories)
        log_probs.update(zip(keys, log_probs_))

    return log_probs


def log_prob_trajectories(env, algorithm):
    def _log_prob(params, net_state, trajectories):
        def _scan_fun(state, actions):
            log_probs, states, partial_trajs, t = state  # Unpack the state

            # Apply the policy network
            observations = env.func_state_to_observation(states, partial_trajs)
            log_pi = algorithm.log_policy(params, net_state, observations)

            # Compute the log forward probability of the action
            log_pF = jnp.take_along_axis(log_pi, actions[:, None], axis=1)
            log_pF = jnp.squeeze(log_pF, axis=1)
            log_probs = jnp.where(actions == -1, log_probs, log_probs + log_pF)

            # Add action to partial trajectories
            partial_trajs = partial_trajs.at[:, t].set(actions)

            # Step in the environment
            states = env.func_step(states, actions)

            return ((log_probs, states, partial_trajs, t + 1), None)

        # Initialize state
        batch_size = trajectories.shape[0]
        log_probs = jnp.zeros((batch_size,), dtype=jnp.float32)
        states = env.func_reset(batch_size)
        partial_trajs = jnp.full_like(trajectories, -1)
        t = jnp.array(0, dtype=jnp.int32)
        init_state = (log_probs, states, partial_trajs, t)

        # Run the actions
        state, _ = jax.lax.scan(_scan_fun, init_state, trajectories.T)

        return state[0]  # log_probs

    return _log_prob
