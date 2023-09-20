import numpy as np
import networkx as nx
import jax

from copy import deepcopy


def get_cache(env, algorithm, params, state, batch_size=256):
    log_policy = jax.jit(algorithm.log_policy)
    cache = dict()

    for keys, observations in env.all_states_batch_iterator(batch_size=batch_size):
        log_probs = log_policy(params, state, observations)
        cache.update(zip(keys, log_probs))
    
    return cache


def push_source_flow_to_terminating_states(mdp_state_graph, cache):
    mdp_state_graph = deepcopy(mdp_state_graph)

    # Initialize log_flow & log_prob to be -np.inf (flow = 0) for all nodes
    nx.set_node_attributes(mdp_state_graph, -np.inf, 'log_flow')
    nx.set_node_attributes(mdp_state_graph, -np.inf, 'log_prob')

    # Except initialize log_flow to be 0 (flow = 1) for source node
    initial_state = mdp_state_graph.graph['initial']
    nx.set_node_attributes(mdp_state_graph, {initial_state: 0}, 'log_flow')

    for state in nx.topological_sort(mdp_state_graph):
        log_flow_incoming = mdp_state_graph.nodes[state]['log_flow']
        log_probs = cache[state]

        for _, child, edge_attr in mdp_state_graph.edges(state, data=True):
            # Get the log-probability of taking the action to get to "child"
            log_prob_action = log_probs[edge_attr['action']]

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
            log_prob_stop = cache[state][-1]

            # Set the log-probability of the terminating state (log_flow + log-probability of terminating)
            log_flow = mdp_state_graph.nodes[state]['log_flow']
            nx.set_node_attributes(
                mdp_state_graph,
                {state: log_flow + log_prob_stop},
                'log_prob'
            )

    return mdp_state_graph


def model_log_posterior(env, algorithm, params, state, batch_size=256):
    cache = get_cache(env, algorithm, params, state, batch_size=batch_size)
    mdp_state_graph = push_source_flow_to_terminating_states(env.mdp_state_graph, cache)

    log_posterior = dict()
    for state, is_terminating in mdp_state_graph.nodes(data='terminating', default=False):
        if is_terminating:
            log_posterior[state] = mdp_state_graph.nodes[state]['log_prob']

    return log_posterior
