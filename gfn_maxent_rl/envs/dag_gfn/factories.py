import chex

from jumanji.wrappers import VmapAutoResetWrapper

from gfn_maxent_rl.envs.dag_gfn.scores import Scorer, ZeroScorer, LinearGaussianScorer, BGeScorer
from gfn_maxent_rl.envs.dag_gfn.graph_priors import GraphPrior, UniformPrior
from gfn_maxent_rl.envs.dag_gfn.env import DAGEnvironment


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


def get_graph_prior(name: str, num_variables: int, **kwargs) -> 'GraphPrior':
    priors = {
        'uniform': UniformPrior,
    }
    if name not in priors:
        valid_priors = ', '.join(priors.keys())
        raise ValueError(f'Unknown graph prior: {name}. Must be one of {{{valid_priors}}}.')
    return priors[name](num_variables, **kwargs)


def get_dag_gfn_env(
    data: chex.Array,
    prior_name: str,
    scorer_name: str,
    prior_kwargs: dict = {},
    scorer_kwargs: dict = {}
) -> 'VmapAutoResetWrapper':
    # Get the graph prior & scorer for reward computation
    num_variables = data.shape[1]
    prior = get_graph_prior(prior_name, num_variables, **prior_kwargs)
    scorer = get_scorer(scorer_name, data, **scorer_kwargs)

    # Create the environment
    env = DAGEnvironment(prior=prior, scorer=scorer)
    env = VmapAutoResetWrapper(env)

    return env
