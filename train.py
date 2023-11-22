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
import datetime

from gfn_maxent_rl.utils.metrics import mean_phd, mean_shd, entropy
from gfn_maxent_rl.utils.exhaustive import exact_log_posterior
from gfn_maxent_rl.utils.async_evaluation import AsyncEvaluator
from gfn_maxent_rl.utils.evaluations import evaluation


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
    time = f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    config.experiment_name = config.exp_name_algorithm + '_' + config.exp_name_env + '_' + time
    run = wandb.init(
        entity='tristandeleu_mila_01',
        project='gfn_maxent_rl',
        group=config.group_name,
        name=config.experiment_name,
        settings=wandb.Settings(start_method='fork'),
        mode=config.upload,
        config=wandb_config
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
    algorithm.optimizer = hydra.utils.instantiate(config.optimizer)
    params, state = algorithm.init(key, replay.dummy_samples)

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - config.exploration.min_exploration),
        transition_steps=config.num_iterations // config.exploration.warmup_prop,
        transition_begin=config.prefill,
    ))

    target = {
        'log_probs': exact_log_posterior(env, batch_size=config.batch_size)
    }
    wandb.summary['target/entropy'] = entropy(target['log_probs'])
    evaluator = AsyncEvaluator(env, algorithm, run, ctx='spawn', target=target)

    observations, _ = env.reset()
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
                # Sample from the replay buffer, and do one step of gradient
                samples = replay.sample(batch_size=config.batch_size, rng=rng)
                params, state, logs = algorithm.step(params, state, samples)

                train_steps = iteration - config.prefill

                if ('graph' in infos) and (iteration % config.evaluation_every == 0):
                    valid_returns, valid_adjacencies = evaluation(params.online, state.network, key,
                                                                  algorithm, env_valid, config)
                    wandb.log({
                        "mean_pairwise_hamming_distance": mean_phd(valid_adjacencies),
                        "mean_structural_hamming_distance": mean_shd(ground_truth, valid_adjacencies),
                        "average_Return": valid_returns,
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
                    'step': train_steps,
                    'loss/actor': logs['actor_loss'].mean().item() if ('actor_loss' in logs) else 0.,
                    'loss/critic': logs['critic_loss'].mean().item() if ('critic_loss' in logs) else 0.,
                })

    evaluator.join()


if __name__ == '__main__':
    main()
