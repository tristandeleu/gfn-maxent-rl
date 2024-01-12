import numpy as np
import jax

from gfn_maxent_rl.utils.beam_search import beam_search_forward


def estimate_log_probs_beam_search(
        env,
        algorithm,
        params,
        net_state,
        samples,
        beam_size=128,
        batch_size=1
):
    log_probs = dict()

    for keys, action_masks, max_length \
            in env.action_masks_batch_iterator(samples, batch_size=batch_size):
        # Vmap beam search over multiple samples
        beam_search = beam_search_forward(env, algorithm, beam_size=beam_size, max_length=max_length)
        beam_search = jax.vmap(beam_search, in_axes=(None, None, 0))
        beam_search = jax.jit(beam_search)

        # Run beam search on the samples
        trajectories, log_probs_, logs = beam_search(params, net_state, action_masks)
        if not np.all(logs['is_valid_length']):
            raise RuntimeError('Some trajectories are longer than the maximum length.')
        if not np.all(logs['is_valid_trajectories']):
            raise RuntimeError('Not all the trajectories lead to the same state.')

        # Store the log-probabilities
        log_probs.update(zip(keys, log_probs_))

    # TODO: Add trajectories sampled using backward policy & rejection sampling,
    # and add correction by the number of trajectories.

    return log_probs
