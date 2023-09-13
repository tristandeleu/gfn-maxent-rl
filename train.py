import jax.numpy as jnp
import numpy as np
import jax
import optax
import wandb
import hydra

from numpy.random import default_rng
from tqdm.auto import trange
from pathlib import Path

from gfn_maxent_rl.envs.dag_gfn.policy import policy_network, q_network
from gfn_maxent_rl.envs.dag_gfn.factories import get_dag_gfn_env
from gfn_maxent_rl.envs.dag_gfn.data_generation.data import load_artifact_continuous
from gfn_maxent_rl.data import ReplayBuffer
from gfn_maxent_rl.algos.detailed_balance import GFNDetailedBalance
from gfn_maxent_rl.algos.soft_actor_critic import SAC


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
    api = wandb.Api()

    # Set the RNGs for reproducibility
    rng = default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Get the artifact from wandb
    artifact = api.artifact(config.artifact)
    artifact_dir = Path(artifact.download()) / f'{config.seed:02d}'
    # wandb.config['data'] = artifact.metadata

    if config.seed not in artifact.metadata['seeds']:
        raise ValueError(f'The seed `{config.seed}` is not in the list of seeds '
            f'for artifact `{config.artifact}`: {artifact.metadata["seeds"]}')

    train, _, _ = load_artifact_continuous(artifact_dir)

    # Create the environment
    env = get_dag_gfn_env(
        data=train,
        prior_name=config.env.prior,
        score_name=config.env.score,
        num_envs=config.env.num_envs
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        capacity=config.replay.capacity,
        num_variables=env.num_variables,
    )

    # Create the algorithm
    # algorithm = GFNDetailedBalance(
    #     network=policy_network,
    #     update_target_every=config.update_target_every,
    # )
    algorithm = SAC(
        actor_network=policy_network,
        critic_network=q_network,
        update_target_every=config.update_target_every,
    )
    algorithm.optimizer = optax.adam(config.lr)
    params, state = algorithm.init(key, replay.dummy_samples)

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - config.exploration.min_exploration),
        transition_steps=config.num_iterations // config.exploration.warmup_prop,
        transition_begin=config.replay.prefill,
    ))

    observations, _ = env.reset()
    with trange(config.replay.prefill + config.num_iterations) as pbar:
        for iteration in pbar:
            epsilon = exploration_schedule(iteration)

            # Sample actions from the model (with exploration)
            actions, key, logs = algorithm.act(
                params.online, state.network, key, observations, epsilon)
            actions = np.asarray(actions)

            # Apply the actions in the environment
            next_observations, rewards, dones, _, _ = env.step(actions)

            # Add the transitions to the replay buffer
            replay.add(observations, actions, rewards, dones, next_observations)
            observations = next_observations

            if (iteration >= config.replay.prefill) and replay.can_sample(config.batch_size):
                # Sample from the replay buffer, and do one step of gradient
                samples = replay.sample(batch_size=config.batch_size, rng=rng)
                params, state, logs = algorithm.step(params, state, samples)

                train_steps = iteration - config.replay.prefill
                # TODO: Logs in wandb

                pbar.set_postfix(loss=f'{logs["loss"]:.3f}')


if __name__ == '__main__':
    main()
