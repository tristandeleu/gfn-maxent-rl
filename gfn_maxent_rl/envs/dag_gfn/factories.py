import chex

from gfn_maxent_rl.envs.dag_gfn.scores import Scorer, ZeroScorer, LinearGaussianScorer, BGeScorer
from gfn_maxent_rl.envs.dag_gfn.graph_priors import Prior, UniformPrior


def get_scorer(name: str, data: chex.Array, **kwargs) -> 'Scorer':
    scorers = {
        'zero': ZeroScorer,
        'lingauss': LinearGaussianScorer,
        'bge': BGeScorer,
    }
    if name not in scorers:
        valid_scorers = ', '.join(scorers.keys())
        raise ValueError(f'Unknown scorer: {name}. Must be one of {{{valid_scorers}}}.')
    return scorers[name](data, **kwargs)


def get_graph_prior(name: str, num_variables: int, **kwargs) -> 'Prior':
    priors = {
        'uniform': UniformPrior,
    }
    if name not in priors:
        valid_priors = ', '.join(priors.keys())
        raise ValueError(f'Unknown graph prior: {name}. Must be one of {{{valid_priors}}}.')
    return priors[name](num_variables, **kwargs)
