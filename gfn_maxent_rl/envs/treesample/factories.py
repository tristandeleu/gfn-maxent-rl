from numpy.random import default_rng

from gfn_maxent_rl.envs.treesample.chain_env import chain_env
from gfn_maxent_rl.envs.treesample.permuted_chain_env import permuted_chain_env
from gfn_maxent_rl.envs.treesample.factor_graph1_env import factor_graph1_env
from gfn_maxent_rl.envs.treesample.factor_graph2_env import factor_graph2_env
from gfn_maxent_rl.envs.treesample.wrappers import FixedOrderingWrapper


def get_treesample_env(
    name,
    num_variables,
    num_categories,
    num_envs=1,
    fixed_ordering=False,
    rng=default_rng(),
    **kwargs
):
    factor_graphs = {
        'chain': (chain_env, {
            'rbf_bandwidth': 1.,
            'rbf_scale': 0.5,
            'factor': 2.5
        }),
        'permuted_chain': (permuted_chain_env, {
            'alpha': 1.
        }),
        'factor_graph1': (factor_graph1_env, {
            'max_retries': 1000,
            'max_clique_size': 4
        }),
        'factor_graph2': (factor_graph2_env, {
            'max_retries': 1000,
            'max_clique_size': 4,
            'factor': 2.
        })
    }
    if name not in factor_graphs:
        raise ValueError(f'Unknown factor graph: {name}')
    
    fn, kwargs = factor_graphs[name]
    env = fn(
        num_envs,
        num_variables,
        num_categories,
        rng=rng,
        **kwargs
    )

    if fixed_ordering:
        env = FixedOrderingWrapper(env)

    infos = {'metadata': kwargs}

    return (env, infos)
