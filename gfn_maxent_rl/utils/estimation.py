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
        bwd_trajectories, log_num_trajectories = env.backward_sample_trajectories(
            keys, num_trajectories, max_length=max_length, blacklist=blacklist, rng=rng)
        
        # Compute the log-probabilities of the backward trajectories
        log_probs_bwd_trajs = log_prob_fn(params, net_state, bwd_trajectories)

        # Average over all trajectories, and offset by total number of trajectories
        # Assumption: the beam-size is smaller than the total number of trajectories
        offset = log_num_trajectories + np.log1p(-np.exp(math.log(beam_size) - log_num_trajectories))
        bwd_log_probs = offset + np.mean(log_probs_bwd_trajs, axis=1)

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
    log_probs = dict()

    # Vmap function over multiple samples
    log_prob_fn = log_prob_trajectories(env, algorithm)
    log_prob_fn = jax.vmap(log_prob_fn, in_axes=(None, None, 0))
    log_prob_fn = jax.jit(log_prob_fn)

    num_batches = math.ceil(len(samples) / batch_size)
    for keys, max_length in tqdm(env.key_batch_iterator(samples, batch_size=batch_size),
            total=num_batches, disable=(not verbose), **kwargs):
        # Sample random trajectories
        trajectories, log_num_trajectories = env.backward_sample_trajectories(
            keys, num_trajectories, max_length=max_length, rng=rng)

        # Compute the log-probabilities of all trajectories
        log_probs_trajs = log_prob_fn(params, net_state, trajectories)
        log_probs_trajs = np.asarray(log_probs_trajs)

        # Average over all trajectories, and offset by total number of trajectories
        log_probs_ = log_num_trajectories + np.mean(log_probs_trajs, axis=1)
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
