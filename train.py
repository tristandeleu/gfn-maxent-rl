import jax.numpy as jnp
import numpy as np
import jax
import optax
import hydra
import networkx as nx

from numpy.random import default_rng
from tqdm.auto import trange

from gfn_maxent_rl.utils.exhaustive import exact_log_posterior
from gfn_maxent_rl.utils.async_evaluation import AsyncEvaluator
from gfn_maxent_rl.envs.errors import StatesEnumerationError


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
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
    env_valid, _ = hydra.utils.instantiate(
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
    params, state = algorithm.init(key)

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - config.exploration.min_exploration),
        transition_steps=config.exploration.warmup,
        transition_begin=config.prefill,
    ))

    target = {}
    try:
        target['log_probs'] = exact_log_posterior(env, batch_size=config.batch_size)
    except StatesEnumerationError:
        pass
    evaluator = AsyncEvaluator(env, algorithm, None, ctx='spawn', target=target)

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

                if train_steps % config.log_every == 0:
                    evaluator.enqueue(
                        params.online,
                        state.network,
                        train_steps,
                        batch_size=config.batch_size,
                    )

                pbar.set_postfix(loss=f'{logs["loss"]:.3f}')

    # Evaluate the final model

    evaluator.enqueue(
        params.online,
        state.network,
        config.num_iterations,
        batch_size=config.batch_size
    )
    metrics = evaluator.join()


if __name__ == '__main__':
    main()
