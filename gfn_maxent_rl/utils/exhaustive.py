import numpy as np
import networkx as nx

from scipy.special import logsumexp
from copy import deepcopy


def compute_cache(env, log_policy, params, state, batch_size=256):
    cache = dict()

    for keys, observations in env.all_states_batch_iterator(batch_size=batch_size):
        log_probs = log_policy(params, state, observations)
        cache.update(zip(keys, np.asarray(log_probs)))
    
    return cache


def push_source_flow_to_terminating_states(mdp_state_graph, cache):
    mdp_state_graph = deepcopy(mdp_state_graph)

    # Get the shape of the cache (in case there are multiple caches)
    dummy_log_probs = next(iter(cache.values()))
    shape = dummy_log_probs.shape[:-1]  # Dimension -1 corresponds to actions

    # Initialize log_flow & log_prob to be -np.inf (flow = 0) for all nodes
    neg_inf = np.full(shape, -np.inf)
    nx.set_node_attributes(mdp_state_graph, neg_inf, 'log_flow')
    nx.set_node_attributes(mdp_state_graph, neg_inf, 'log_prob')

    # Except initialize log_flow to be 0 (flow = 1) for source node
    initial_state = mdp_state_graph.graph['initial']
    nx.set_node_attributes(
        mdp_state_graph,
        {initial_state: np.zeros(shape)},
        'log_flow'
    )

    for state in nx.topological_sort(mdp_state_graph):
        log_flow_incoming = mdp_state_graph.nodes[state]['log_flow']
        log_probs = cache[state]

        for _, child, action in mdp_state_graph.edges(state, data='action'):
            # Get the log-probability of taking the action to get to "child"
            log_prob_action = log_probs[..., action]

            # Update the log-flow of the child
            existing_log_flow_child = mdp_state_graph.nodes[child]['log_flow']
            updated_log_flow_child = np.logaddexp(
                existing_log_flow_child,
                log_flow_incoming + log_prob_action
            )
            nx.set_node_attributes(
                mdp_state_graph,
                {child: updated_log_flow_child},
                'log_flow'
            )

    # Add the probability of the stop action for terminating states
    for state, is_terminating in mdp_state_graph.nodes(data='terminating', default=False):
        if is_terminating:
            # Get the log-probability of the stop action (-1 = stop action)
            log_prob_stop = cache[state][..., -1]

            # Set the log-probability of the terminating state (log_flow + log-probability of terminating)
            log_flow = mdp_state_graph.nodes[state]['log_flow']
            nx.set_node_attributes(
                mdp_state_graph,
                {state: log_flow + log_prob_stop},
                'log_prob'
            )

    return mdp_state_graph


def exact_log_posterior(env, batch_size=256):
    log_posterior = dict()

    for keys, observations in env.all_states_batch_iterator(batch_size, terminating=True):
        log_rewards = env.log_reward(observations)
        log_posterior.update(zip(keys, log_rewards))

    # Compute the log-partition function
    log_rewards = np.asarray(list(log_posterior.values()))
    log_Z = logsumexp(log_rewards)

    # Normalize the rewards
    for key, log_reward in log_posterior.items():
        log_posterior[key] = log_reward - log_Z

    return log_posterior
