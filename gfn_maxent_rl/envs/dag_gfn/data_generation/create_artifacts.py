import os
import wandb
import math
import pickle
import shutil
import networkx as nx
import numpy as np

from tqdm.auto import tqdm
from pathlib import Path
from numpy.random import default_rng

from gfn_maxent_rl.envs.dag_gfn.data_generation.graphs import sample_erdos_renyi_graph, sample_linear_gaussian
from gfn_maxent_rl.envs.dag_gfn.data_generation.data import sample_from_linear_gaussian

# PYTHONPATH=. python gfn_maxent_rl/envs/dag_gfn/data_generation/create_artifacts.py \
#     --num_variables 10 \
#     --num_edges_per_node 2 \
#     --num_train_samples 100 \
#     --num_valid_samples 100 \
#     --num_seeds 5


def generate_artifact(args, root, seeds):
    # Get default kwargs for CPDs
    cpd_kwargs = {
        'loc_edges': 0.,
        'scale_edges': 1.,
        'obs_scale': math.sqrt(0.1)
    }

    # Create the artifact name
    artifact_name = 'er{num_edges_per_node}-lingauss-d{num_vars:03d}'.format(
        num_edges_per_node=args.num_edges_per_node,
        num_vars=args.num_variables
    )

    # Create a temporary folder
    root = root / artifact_name
    root.mkdir(exist_ok=True, parents=True)

    # Generate 
    for seed in tqdm(seeds, desc=f'Generate artifact {artifact_name}'):
        rng = default_rng(seed)

        # Generate the graph
        graph = sample_erdos_renyi_graph(
            args.num_variables,
            num_edges_per_node=args.num_edges_per_node,
            rng=rng,
        )
        # Create linear-Gaussian CPDs
        graph = sample_linear_gaussian(graph, rng=rng, **cpd_kwargs)

        # Verify that the graph if a DAG
        if not nx.is_directed_acyclic_graph(graph):
            raise RuntimeError('The graph is not acyclic.')

        # Sample data
        train_data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_train_samples,
            rng=rng
        )
        valid_data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_valid_samples,
            rng=rng
        )

        # Save the data
        folder = root / f'{seed:02d}'
        folder.mkdir(exist_ok=True)

        with open(folder / 'graph.pkl', 'wb') as f:
            pickle.dump(graph, f)
        
        adjacency = nx.to_numpy_array(graph, weight=None)
        with open(folder / 'adjacency.npy', 'wb') as f:
            np.save(f, adjacency)

        train_data.to_csv(folder / 'train_data.csv')
        valid_data.to_csv(folder / 'valid_data.csv')
    
    # Create the artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type='dataset',
        metadata={
            'graph_type': 'erdos_renyi',
            'graph_kwargs': {},
            'cpd_type': 'lingauss',
            'cpd_kwargs': cpd_kwargs,
            'num_variables': args.num_variables,
            'num_edges_per_node': args.num_edges_per_node,
            'num_samples': {
                'train': args.num_train_samples,
                'valid': args.num_valid_samples
            },
            'seeds': seeds,
            'ground_truth': True  # Ground truth graph is available
        }
    )
    artifact.add_dir(root)
    return artifact


def main(args):
    # Create a temporary folder
    root = Path(os.getenv('SLURM_TMPDIR', '.'), 'data', 'artifacts')
    root.mkdir(exist_ok=True, parents=True)

    # Create seeds
    if args.seeds is None:
        if args.num_seeds is None:
            raise ValueError('Either argument `--seeds` or `--num_seeds` '
                             'must be set.')
        seeds = list(range(args.num_seeds))
    else:
        seeds = list(args.seeds)

    try:
        # Create all the artifacts
        artifact = generate_artifact(args, root, seeds)

        # Upload the artifact
        if args.upload:
            with wandb.init(
                    project='gfn_maxent_rl',
                    entity='tristandeleu_mila_01',
                    group='data_generation',
                    config=args,
                    settings=wandb.Settings(start_method='fork')
                ) as run:

                run.log_artifact(artifact)
        else:
            print('All the artifacts were properly generated. You can upload them '
                'by running the same command with the argument `--upload`.')
    finally:
        # Delete temporary folder
        shutil.rmtree(root)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(description='Data generation and export to W&B.')

    # Graph
    graph = parser.add_argument_group('Graph')
    graph.add_argument('--num_variables', type=int, required=True,
        help='Number of variables')
    graph.add_argument('--num_edges_per_node', type=int, required=True,
        help='Average number of edges per node')

    # Data
    data = parser.add_argument_group('Data')
    data.add_argument('--num_train_samples', type=int, required=True,
        help='Number of training samples')
    data.add_argument('--num_valid_samples', type=int, required=True,
        help='Number of validation samples')
    
    # Misc
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--seeds', type=json.loads, default=None,
        help='List of random seeds')
    misc.add_argument('--num_seeds', type=int, default=None,
        help='Number of random seeds')
    misc.add_argument('--upload', action='store_true',
        help='Upload the artifacts to wandb')

    args = parser.parse_args()

    main(args)