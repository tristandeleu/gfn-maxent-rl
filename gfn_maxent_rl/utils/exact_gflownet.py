import numpy as np
import networkx as nx
import jax

from collections import defaultdict, deque

from gfn_maxent_rl.envs.dag_gfn.jraph_utils import to_graphs_tuple
from gfn_maxent_rl.utils.exhaustive import all_dags_batch_iterator
from gfn_maxent_rl.utils.graphs import compute_masks


def get_gflownet_cache(gflownet, params, state, nodelist, batch_size=256):
    """Cache the results of the GFlowNet for all the states.

    This function caches the log-probabilities for all the actions and for
    all the states of the GFlowNet.

    Parameters
    ----------
    gflownet : DAGGFlowNet instance
        The GFlowNet class, containing the definition of the model.

    params : GFNParameters instance
        The parameters of the GFlowNet (i.e., the parameters of the model, as
        well as log(Z)).

    state : TODO

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    batch_size : int
        The batch-size of the inputs of the GFlowNet.

    Returns
    -------
    cache : dict of (frozenset, {'mask': np.ndarray, 'log_pi': np.ndarray})
        The cache of log-probabilities returned by the GFlowNet. The keys of
        the cache are the graphs (encoded as a frozenset of their edges), and
        the corresponding value is an array of size `(num_variables ** 2 + 1,)`
        containing the log-probabilities of all the actions in that state
        (including the "stop" action, at the last index).
    """
    log_policy = jax.jit(gflownet.log_policy)
    cache = dict()

    for adjacencies in all_dags_batch_iterator(len(nodelist), batch_size=batch_size):
        masks = compute_masks(adjacencies)
        observations = {
            'adjacency': adjacencies.astype(np.float32),
            'mask': masks.astype(np.float32),
            'graph': to_graphs_tuple(adjacencies),
        }
        log_pis = log_policy(params, state, observations)

        for adjacency, mask, log_pi in zip(adjacencies, masks, np.asarray(log_pis)):
            sources, targets = np.nonzero(adjacency)
            edges = frozenset((nodelist[i], nodelist[j])
                for (i, j) in zip(sources, targets))
            cache[edges] = {'mask': mask, 'log_pi': log_pi}

    return cache


def get_children(graph, gfn_cache, nodelist):
    """Gets all the children of a graph.

    This function returns a set of the next states, from a particular state
    `graph`, with its corresponding log-probability. Note that the set of
    children includes the stop action, encoded as a `None` action, for which
    the child graph is the same as the current graph.

    Parameters
    ----------
    graph : nx.DiGraph instance
        The current graph.

    gfn_cache : dict
        The cache of log-probabilities returned by the GFlowNet. See
        `dag_gflownet.utils.gflownet.get_gflownet_cache` for details.

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    Returns
    -------
    children : set of tuples
        The set of all the next state from the current graph, with their
        corresponding log-probability. Each child is represented as
        `(next_graph, action, log_prob)`, where `next_graph` is a nx.DiGraph
        instance, `action` is the edge added (as a tuple of nodes), and
        `log_prob` is the log-probability of this action. Not that the "stop"
        action is encoded as the action `None`.
    """
    num_variables = len(nodelist)
    cache = gfn_cache[frozenset(graph.edges())]
    mask, log_pi = cache['mask'], cache['log_pi']

    children = {(graph, None, log_pi[-1])}  # The stop action
    for action in np.nonzero(mask[:-1]):  # Non-stop actions
        source, target = divmod(action, num_variables)
        source, target = nodelist[source], nodelist[target]

        next_graph = graph.copy()
        next_graph.add_edge(source, target)

        children.add((next_graph, (source, target), log_pi[action]))

    return children


def push_source_flow_to_terminal_states(gfn_state_graph, source_state_graph):
    """Compute a hashable key for a graph.

    This function traverses the GFlowNet state-action space graph (DAG) in a 
    topologically sorted order and "pushes" the log_flow from each node to
    its children according to the log_prob_action specified on the edges.
    The topological sort ensures that all the flow has "arrived" at a node
    before "moving" its flow to its children.

    Parameters
    ----------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space where each node represents one GFlowNet
        state and each edge represents one GFlowNet action.

    source_state_graph: nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space but now each node has an attribute
        named log_flow, which is -np.inf for non-terminal states and
        the marginal log probability for the terminal states.
    """
    # Initialize log_flow to be -np.inf (flow = 0) for all nodes
    nx.set_node_attributes(gfn_state_graph, -np.inf, "log_flow")

    # Except initialize log_flow to be 0 (flow = 1) for source node
    source_node_key = frozenset(source_state_graph.edges)
    nx.set_node_attributes(gfn_state_graph, {source_node_key: 0}, "log_flow")

    # Push flow through sorted graph
    for state in nx.topological_sort(gfn_state_graph):
        current_node = gfn_state_graph.nodes[state]
        log_flow_incoming = current_node["log_flow"]  # log_flow is log probability of reaching this node starting from source node

        # Compute terminal_log_flow
        stop_action_log_flow = current_node["stop_action_log_flow"]  # probability of taking stop action from this node

        # terminal prob = incoming probability * p(stop action at this node)
        current_node["terminal_log_flow"] = log_flow_incoming + stop_action_log_flow

        # Push flow along edges to children
        edges = gfn_state_graph.edges(state, data=True)
        for _, child, edge_attr in edges:
            log_prob_action = edge_attr["log_prob_action"]
            existing_log_flow_child = gfn_state_graph.nodes[child]["log_flow"]
            updated_log_flow_child = np.logaddexp(
                existing_log_flow_child, log_flow_incoming + log_prob_action
            )
            nx.set_node_attributes(gfn_state_graph, {child: updated_log_flow_child}, "log_flow")

    return gfn_state_graph


def construct_state_dag_with_bfs(gflownet_cache, nodelist, source_graph=None):
    """Constructs the state-action space of the GFlowNet.

    This function performs Breadth-First Search on the GFlowNet state-action space
    starting from the source state, in order to construct a networkx.DiGraph object
    where each node is a GFlowNet state and each edge is labeled with the action
    and the log probability of taking that action. Each node is also labeled with
    the stop_action_log_flow which contains the probability of terminating at that state.

    Parameters
    ----------
    gflownet_cache :

    nodelist :

    source_graph : nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space.

    source_graph : nx.DiGraph instance
        The graph representing the source state.
    """
    gfn_state_graph = nx.DiGraph()
    is_state_queued = defaultdict(bool)
    states_to_visit = deque()

    if source_graph is None:
        source_graph = nx.DiGraph()
        source_graph.add_nodes_from(nodelist)
    source_graph_key = frozenset(source_graph.edges)

    gfn_state_graph.add_node(source_graph_key, graph=source_graph)
    states_to_visit.append(source_graph)
    is_state_queued[source_graph_key] = True
    while len(states_to_visit) > 0:
        current_graph = states_to_visit.popleft()
        current_graph_key = frozenset(current_graph.edges)
        children = get_children(current_graph, gflownet_cache, nodelist)
        for child_graph, action, log_prob in children:
            if action is None:  # stop action
                # Encode the stop action as a node attribute
                gfn_state_graph.nodes[current_graph_key]['stop_action_log_flow'] = log_prob
            else:
                child_graph_key = frozenset(child_graph.edges)
                if child_graph_key not in gfn_state_graph:
                    gfn_state_graph.add_node(child_graph_key, graph=child_graph)
                gfn_state_graph.add_edge(
                    current_graph_key,
                    child_graph_key,
                    action=action,
                    log_prob_action=log_prob
                )
                already_visited = is_state_queued[child_graph_key]
                if not already_visited:
                    states_to_visit.append(child_graph)
                    is_state_queued[child_graph_key] = True

    return gfn_state_graph, source_graph


def get_posterior_approximate(gflownet, params, state, nodelist, batch_size=256):
    gfn_cache = get_gflownet_cache(gflownet, params, state, nodelist, batch_size)
    gfn_state_graph, source_state_graph = construct_state_dag_with_bfs(
        gfn_cache, nodelist)
    gfn_state_graph = push_source_flow_to_terminal_states(
        gfn_state_graph, source_state_graph)
    return gfn_state_graph
