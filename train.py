import jax.numpy as jnp
import numpy as np
import jax
import optax

from numpy.random import default_rng
from tqdm.auto import trange

from gfn_maxent_rl.envs.dag_gfn.policy import policy_network
from gfn_maxent_rl.envs.dag_gfn.factories import get_dag_gfn_env
from gfn_maxent_rl.data import ReplayBuffer
from gfn_maxent_rl.algos.detailed_balance import GFNDetailedBalance


def main(args):
    # Set the RNGs for reproducibility
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Generate dummy data (for tests only). TODO: Load proper data
    data = jax.random.normal(key, shape=(100, 5))

    # Create the environment
    env = get_dag_gfn_env(
        data=data,
        prior_name=args.prior,
        score_name=args.score,
        num_envs=args.num_envs
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        capacity=args.replay_capacity,
        num_variables=env.num_variables,
    )

    # Create the algorithm
    algorithm = GFNDetailedBalance(
        network=policy_network,
        update_target_every=args.update_target_every,
    )
    algorithm.optimizer = optax.adam(args.lr)
    params, state = algorithm.init(key, replay.dummy_samples)

    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(0.),
        end_value=jnp.array(1. - args.min_exploration),
        transition_steps=args.num_iterations // args.exploration_warmup_prop,
        transition_begin=args.prefill,
    ))

    observations, _ = env.reset()
    with trange(args.prefill + args.num_iterations) as pbar:
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

            if (iteration >= args.prefill) and replay.can_sample(args.batch_size):
                # Sample from the replay buffer, and do one step of gradient
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = algorithm.step(params, state, samples)

                train_steps = iteration - args.prefill
                # TODO: Logs in wandb

                pbar.set_postfix(loss=f'{logs["loss"]:.3f}')


if __name__ == '__main__':
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser(description='Comparison between GFlowNets & Maximum Entropy RL.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
        help='Arguments of the prior over graphs.')
    environment.add_argument('--score', type=str, default='zero',
        choices=['zero', 'lingauss', 'bge'],
        help='Score to compute the log-marginal likelihood (default: %(default)s)')
    environment.add_argument('--score_kwargs', type=json.loads, default='{}',
        help='Arguments of the score.')

    # # Data
    # data = parser.add_argument_group('Data')
    # data.add_argument('--artifact', type=str, required=True,
    #     help='Path to the artifact for input data in Wandb')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=128,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
        help='Number of iterations (default: %(default)s)')
    optimization.add_argument('--update_target_every', type=int, default=0,
        help='Frequency of update for the target network (0 = no target network)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')

    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--exploration_warmup_prop', type=int, default=2,
        help='Proportion of training steps for warming up exploration (default: %(default)s)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    misc.add_argument('--log_every', type=int, default=50,
        help='Frequency for logging (default: %(default)s)')
    misc.add_argument('--group_name', type=str, default='default',
        help='Name of the group for Wandb (default: %(default)s)')

    args = parser.parse_args()

    main(args)
