import wandb

from pathlib import Path

from gfn_maxent_rl.envs.dag_gfn.scores import ZeroScore, LinearGaussianScore, BGeScore
from gfn_maxent_rl.envs.dag_gfn.graph_priors import UniformPrior
from gfn_maxent_rl.envs.dag_gfn.env import DAGEnvironment
from gfn_maxent_rl.envs.dag_gfn.base import JointModel
from gfn_maxent_rl.envs.dag_gfn.data_generation.data import load_artifact_continuous


def get_marginal_likelihood(name, data, **kwargs):
    scores = {
        'zero': ZeroScore,
        'lingauss': LinearGaussianScore,
        'bge': BGeScore,
    }
    if name not in scores:
        valid_scores = ', '.join(scores.keys())
        raise ValueError(f'Unknown score (marginal likelihood): {name}. Must be one of {{{valid_scores}}}.')
    return scores[name](data, **kwargs)


def get_graph_prior(name, num_variables, **kwargs):
    priors = {
        'uniform': UniformPrior,
    }
    if name not in priors:
        valid_priors = ', '.join(priors.keys())
        raise ValueError(f'Unknown graph prior: {name}. Must be one of {{{valid_priors}}}.')
    return priors[name](num_variables, **kwargs)


def get_dag_gfn_env(
    artifact,
    prior_name,
    score_name,
    seed=0,
    num_envs=1,
    prior_kwargs={},
    score_kwargs={},
    **kwargs
):
    api = wandb.Api()

    # Get the artifact from Wandb
    artifact = api.artifact(artifact)
    artifact_dir = Path(artifact.download()) / f'{seed:02d}'

    if seed not in artifact.metadata['seeds']:
        raise ValueError(f'The seed `{seed}` is not in the list of seeds '
            f'for artifact `{artifact}`: {artifact.metadata["seeds"]}')

    train, valid, graph = load_artifact_continuous(artifact_dir)

    # Get the graph prior & marginal likelihood for reward computation
    num_variables = train.shape[1]
    graph_prior = get_graph_prior(prior_name, num_variables, **prior_kwargs)
    marginal_likelihood = get_marginal_likelihood(score_name, train, **score_kwargs)
    joint_model = JointModel(
        graph_prior=graph_prior,
        marginal_likelihood=marginal_likelihood
    )

    # Create the environment
    env = DAGEnvironment(num_envs=num_envs, joint_model=joint_model)

    infos = {
        'metadata': artifact.metadata,
        'train': train,
        'valid': valid,
        'graph': graph,
    }

    return (env, infos)
