import json

from pathlib import Path

from gfn_maxent_rl.envs.phylo_gfn.rewards import ExponentialReward
from gfn_maxent_rl.envs.phylo_gfn.env import PhyloTreeEnvironment


DATA_FOLDER = Path(__file__).resolve().parent / 'datasets'
CONFIGS = {
    'DS1': ('DNA_WITH_GAP', 5800.0, 4.),
    'DS2': ('DNA_WITH_GAP', 8000.0, 4.),
    'DS3': ('DNA_WITH_GAP', 8800.0, 4.),
    'DS4': ('DNA_WITH_GAP', 3500.0, 4.),
    'DS5': ('DNA_WITH_GAP', 2300.0, 4.),
    'DS6': ('DNA_WITH_GAP', 2300.0, 4.),
    'DS7': ('DNA_WITH_GAP', 12500.0, 4.),
    'DS8': ('DNA_WITH_GAP', 2800.0, 4.),
}

def get_phylo_gfn_env(
    dataset_name,
    num_envs=1,
    data_folder=DATA_FOLDER,
    **kwargs
):
    if dataset_name not in CONFIGS:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    with open(data_folder / f'{dataset_name}.json', 'r') as f:
        sequences = json.load(f)

    sequence_type, C, scale = CONFIGS[dataset_name]

    reward = ExponentialReward(num_nodes=len(sequences), scale=scale, C=C)

    env = PhyloTreeEnvironment(
        num_envs=num_envs,
        sequences=sequences,
        reward=reward,
        sequence_type=sequence_type
    )

    return (env, CONFIGS[dataset_name])
