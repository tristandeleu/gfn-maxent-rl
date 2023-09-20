import jax.numpy as jnp
import numpy as np
import jax
import optax
import wandb
import hydra
import omegaconf
import networkx as nx

from numpy.random import default_rng
from tqdm.auto import trange

from gfn_maxent_rl.utils.metrics import mean_phd, mean_shd
from gfn_maxent_rl.utils.exhaustive import exact_log_posterior
from gfn_maxent_rl.utils.async_evaluation import AsyncEvaluator


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    run = wandb.init(
        entity='tristandeleu_mila_01',
        project='gfn_maxent_rl',
        group=config.group_name,
        settings=wandb.Settings(start_method='fork'),
        mode=config.upload
    )

    # Set the RNGs for reproducibility
    rng = default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Create the environment
    # Train environment
    env, infos = hydra.utils.instantiate(
        config.env,
        num_envs=config.num_envs,
        seed=config.seed,
        rng=rng,
    )
    # Evaluation environment
    env_valid, infos_valid = hydra.utils.instantiate(
        config.env,
        num_envs=config.num_envs,
        seed=config.seed,
        rng=rng,
    )

    if 'graph' in infos:
        ground_truth = nx.to_numpy_array(infos['graph'], weight=None)

    # Add wrapper to the environment
    if config.reward_correction:
        env = hydra.utils.instantiate(config.env_wrapper, env=env)
        env_valid = hydra.utils.instantiate(config.env_wrapper, env=env_valid)

    # Create the replay buffer
    replay = hydra.utils.instantiate(config.replay, env=env)

    # Create the algorithm
    algorithm = hydra.utils.instantiate(config.algorithm, env=env)
    algorithm.optimizer = optax.adam(config.lr)
    params, state = algorithm.init(key, replay.dummy_samples)

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - config.exploration.min_exploration),
        transition_steps=config.num_iterations // config.exploration.warmup_prop,
        transition_begin=config.prefill,
    ))

    evaluator = AsyncEvaluator(env, algorithm, run, ctx='spawn', target={
        'log_probs': exact_log_posterior(env, batch_size=config.batch_size)
    })

    observations, _ = env.reset()
    observations_valid, _ = env_valid.reset()
    indices = None
    with trange(config.prefill + config.num_iterations) as pbar:
        for iteration in pbar:
            epsilon = exploration_schedule(iteration)

            # Sample actions from the model (with exploration)
            actions, key, logs = algorithm.act(
                params.online, state.network, key, observations, epsilon)
            actions = np.asarray(actions)

            # Apply the actions in the environment
            next_observations, rewards, dones, _, _ = env.step(actions)

            # Add the transitions to the replay buffer
            indices = replay.add(observations, actions, rewards, dones, next_observations, indices=indices)
            observations = next_observations

            if (iteration >= config.prefill) and replay.can_sample(config.batch_size):
                returns_array = np.zeros((8))
                dones_array = np.zeros((8))
                # Evaluation samples
                while dones_array.sum() < 100:
                    # Sample actions from the model (w/o exploration)
                    actions_valid, key, _ = algorithm.act(
                        params.online, state.network, key, observations_valid, epsilon=1.)
                    actions_valid = np.asarray(actions_valid)
                    # Apply the actions in the evaluation environment
                    _, rewards_valid, dones_valid, _, _ = env_valid.step(actions_valid)
                    returns_array = np.vstack((returns_array, rewards_valid))
                    dones_array = np.vstack((dones_array, 1 * dones_valid))
                    import pdb; pdb.set_trace()

                stop_dones_index = dones_array.shape[0] - np.argmax(dones_array[::-1], axis=0) - 1
                dones_array_corrected = np.zeros_like(dones_array)
                for column, row in enumerate(stop_dones_index):
                    dones_array_corrected[:row+1, column] = 1
                terminated_rewards = returns_array * dones_array_corrected
                average_Return = terminated_rewards.sum() / dones_array.sum()



                # Sample from the replay buffer, and do one step of gradient
                samples = replay.sample(batch_size=config.batch_size, rng=rng)
                params, state, logs = algorithm.step(params, state, samples)

                train_steps = iteration - config.prefill

                if ('graph' in infos) and ('observation' in samples):
                    adjacencies = samples['observation']['adjacency']
                    wandb.log({
                        "mean_pairwise_hamming_distance": mean_phd(adjacencies),
                        "mean_structural_hamming_distance": mean_shd(ground_truth, adjacencies),
                        'step': train_steps,
                    }, commit=False)

                if train_steps % config.log_every == 0:
                    evaluator.enqueue(
                        params.online,
                        state.network,
                        train_steps,
                        batch_size=config.batch_size,
                    )

                pbar.set_postfix(loss=f'{logs["loss"]:.3f}')
                wandb.log({
                    'loss': logs["loss"].item(),
                    "average_Return": average_Return,
                    'step': train_steps
                })

    evaluator.join()


if __name__ == '__main__':
    main()
