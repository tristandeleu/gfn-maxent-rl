import jax.numpy as jnp
import jax


def beam_search_forward(env, algorithm, beam_size=1):
    def _beam_search(params, net_state, action_mask):
        # action_mask: (action_dim,), does not include stop "action"
        # i.e., we have the guarantee that action_mask[-1] == False
        num_actions = action_mask.shape[0]
        length = jnp.sum(action_mask, axis=0)

        def cond_fun(state):
            *_, t = state
            return (t < length)

        def body_fun(state):
            trajectories, log_probs, states, t = state

            # Apply the policy network & mask invalid actions
            observations = env.func_state_to_observation(states, trajectories)
            log_pi = algorithm.log_policy(params, net_state, observations)
            log_pi = jnp.where(action_mask, log_pi, -jnp.inf)

            # Find the "beam_size" best scores
            scores = log_pi + log_probs[:, None]
            log_probs, indices = jax.lax.top_k(scores.reshape(-1), beam_size)

            # Reorder the beams and add actions
            beams, actions = jnp.divmod(indices, log_pi.shape[1])  # Get the actions
            trajectories = trajectories[beams].at[:, t].set(actions)

            # Reorder the beams & step in the environment
            states = jax.tree_util.tree_map(lambda arr: arr[beams], states)
            states = env.func_step(states, actions)

            return (trajectories, log_probs, states, t + 1)

        # Initialization
        trajectories = jnp.full((beam_size, env.max_length), -1, dtype=jnp.int32)
        log_probs = jnp.full((beam_size,), -jnp.inf).at[0].set(0.)
        states = env.func_reset(beam_size)
        t = jnp.array(0, dtype=jnp.int32)

        # Run beam search
        init_state = (trajectories, log_probs, states, t)
        trajectories, log_probs, states, t = jax.lax.while_loop(cond_fun, body_fun, init_state)

        # Add the "stop" action at the end
        observations = env.func_state_to_observation(states, trajectories)
        log_pi = algorithm.log_policy(params, net_state, observations)
        log_probs = log_probs + log_pi[:, -1]  # Add log-probability of the "stop" action
        trajectories = trajectories.at[:, t].set(num_actions)  # Add "stop" action to trajectories

        return (trajectories, log_probs)
    return _beam_search
