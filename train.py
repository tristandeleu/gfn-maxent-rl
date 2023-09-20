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

from gfn_maxent_rl.utils.metrics import mean_phd, mean_shd, jensen_shannon_divergence
from gfn_maxent_rl.utils.exhaustive import exact_log_posterior, model_log_posterior, compute_cache


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
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
    env, infos = hydra.utils.instantiate(
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

    # Compute the target distribution
    log_probs_target = exact_log_posterior(env, batch_size=config.batch_size)
    log_policy = jax.jit(algorithm.log_policy)

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

                if ('graph' in infos) and ('observation' in samples):
                    adjacencies = samples['observation']['adjacency']
                    wandb.log({
                        "mean_pairwise_hamming_distance": mean_phd(adjacencies),
                        "mean_structural_hamming_distance": mean_shd(ground_truth, adjacencies)
                    }, commit=False)

                if train_steps % config.log_every == 0:
                    # Compute the distribution induced by the model
                    cache = compute_cache(
                        env, log_policy, params.online, state.network, batch_size=config.batch_size)
                    # log_probs_model = model_log_posterior(
                    #     env, algorithm, params.online, state.network, batch_size=config.batch_size)

                    # wandb.log({
                    #     'metrics/jsd': jensen_shannon_divergence(log_probs_model, log_probs_target),
                    # }, commit=False)

                pbar.set_postfix(loss=f'{logs["loss"]:.3f}')
                wandb.log({"loss": logs["loss"].item()})


if __name__ == '__main__':
    main()
