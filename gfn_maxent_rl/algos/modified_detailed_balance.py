import jax.numpy as jnp
import optax

from gfn_maxent_rl.algos.base import GFNBaseAlgorithm


class ModifiedDetailedBalance(GFNBaseAlgorithm):
    r"""Modified Detailed Balance loss [1].

    Modification of the Detailed Balance loss, adapted to Markov Decision
    Processes where all the states are terminating (i.e., valid samples
    of the target distribution). The residual can be written as

        \Delta(s, s') = \log \frac{R(s')P_B(s | s')P_F(s_f | s)}{R(s)P_F(s' | s)P_F(s_f | s')}

    References
    ----------
    [1] Tristan Deleu, Antonio Gois, Chris Emezue, Mansi Rankawat, Simon Lacoste-Julien,
        Stefan Bauer, and Yoshua Bengio. Bayesian Structure Learning with
        Generative Flow Networks. Uncertainty in Artificial Intelligence, 2022.
    """
    def loss(self, online_params, target_params, state, samples):
        # Get log P_F(. | s_t) for the current state
        log_pi_t, _ = self.network.apply(
            online_params.network, state, samples['observation'])

        # Get log P_F(. | s_t+1) for the next state
        params = target_params if self.use_target else online_params
        log_pi_tp1, _ = self.network.apply(
            params.network, state, samples['next_observation'])

        # Compute the (modified) detailed balance loss
        log_pF = jnp.take_along_axis(log_pi_t, samples['action'], axis=-1)
        log_pF = jnp.squeeze(log_pF, axis=-1)
        log_pB = -jnp.log(self.env.num_parents(samples['next_observation']))  # Uniform p_B

        # Recall that `samples['reward']` contains the delta-scores: log R(s_t+1) - log R(s_t)
        delta_scores = jnp.squeeze(samples['reward'], axis=1)
        errors = (delta_scores + log_pB + log_pi_t[:, -1] - log_pF - log_pi_tp1[:, -1])
        loss = jnp.mean(optax.huber_loss(errors, delta=1.))

        logs = {'errors': errors, 'loss': loss}
        return (loss, logs)
