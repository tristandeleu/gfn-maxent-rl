import json

from pathlib import Path

from gfn_maxent_rl.envs.phylo_gfn.rewards import ExponentialReward
from gfn_maxent_rl.envs.phylo_gfn.env import PhyloTreeEnvironment


DATA_FOLDER = Path(__file__).resolve().parent / 'datasets'
SEQUENCE_TYPES = {
    'DS1': 'DNA_WITH_GAP',
    'DS2': 'DNA_WITH_GAP',
    'DS3': 'DNA',
    'DS4': 'DNA_WITH_GAP',
    'DS5': 'DNA_WITH_GAP',
    'DS6': 'DNA_WITH_GAP',
    'DS7': 'DNA',
    'DS8': 'DNA_WITH_GAP'
}

def get_phylo_gfn_env(
    dataset_name,
    num_envs=1,
    data_folder=DATA_FOLDER,
    **kwargs
):
    if dataset_name not in SEQUENCE_TYPES:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    with open(data_folder / f'{dataset_name}.json', 'r') as f:
        sequences = json.load(f)

    reward = ExponentialReward(
        num_nodes=len(sequences),
        scale=1.,
        C=0.
    )

    env = PhyloTreeEnvironment(
        num_envs=num_envs,
        sequences=sequences,
        reward=reward,
        sequence_type=SEQUENCE_TYPES[dataset_name]
    )

    return (env, {})
