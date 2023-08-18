import jax.numpy as jnp
import jax

from gfn_maxent_rl.envs.dag_gfn import DAGEnvironment, RewardCorrection
from gfn_maxent_rl.envs.dag_gfn.graph_priors import UniformPrior

def uniform_log_policy(mask):
    log_p_stop = -jnp.log1p(jnp.sum(mask))
    log_p_continue = jnp.where(mask.reshape(-1), log_p_stop, -jnp.inf)
    return jnp.hstack((log_p_continue, log_p_stop))

@jax.jit
def step(state, timestep, key):
    # Sample random action
    log_pi = uniform_log_policy(timestep.observation.mask)
    key, subkey = jax.random.split(key)
    action = jax.random.categorical(subkey, logits=log_pi)
    
    # Step in the environment
    state, timestep = env.step(state, action)

    return (state, timestep, key)

env = DAGEnvironment(prior=UniformPrior(num_variables=5))
env = RewardCorrection(env, alpha=1.)
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
state, timestep = env.reset(subkey)
for _ in range(1000):
    state, timestep, key = step(state, timestep, key)
    assert timestep.extras['is_valid_action']

print(state)
