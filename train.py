import jax.numpy as jnp
import numpy as np
import jax
import optax
import wandb
import hydra

from numpy.random import default_rng
from tqdm.auto import trange


@hydra.main(version_base=None, config_path='config', config_name='default')
def main(config):
    # Set the RNGs for reproducibility
    rng = default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Create the environment
    env, _ = hydra.utils.instantiate(
        config.env,
        num_envs=config.num_envs,
        seed=config.seed,
    )

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
                # TODO: Logs in wandb

                pbar.set_postfix(loss=f'{logs["loss"]:.3f}')


if __name__ == '__main__':
    main()
