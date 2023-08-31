import jax.numpy as jnp
import jax
import jumanji
import chex

from jumanji import types, specs
from typing import Tuple

from gfn_maxent_rl.envs.dag_gfn.types import DAGState, DAGObservation
from gfn_maxent_rl.envs.dag_gfn.scores import ZeroScorer


class DAGEnvironment(jumanji.Environment[DAGState]):
    """Environment for DAG-GFlowNet.

    This is a deterministic environment over the state space of DAGs over d
    nodes, and where the actions are the d^2 possible edges one can add, with
    an additional "stop" action. The transitions in this MDP have the form
    G -> G', where G' is the result of adding a single edge to G. The reward
    for such a transition is the "delta-score", computed as

        r(G, G') = (log P(D | G') + log P(G')) - (log P(D | G) + log P(G))

    The reward is specified by the `prior` and `scorer` arguments.

    Parameters
    ----------
    prior : `GraphPrior` instance
        The prior over graphs P(G). Note that this contains the number of
        nodes d in the DAGs of the environment.

    scorer : `Scorer` instance
        The marginal likelihood P(D | G).
    """
    def __init__(self, prior, scorer=ZeroScorer(data=None)):
        self.prior = prior
        self.scorer = scorer
        self.num_nodes = prior.num_variables

    def reset(self, key: chex.PRNGKey) -> Tuple[DAGState, types.TimeStep[DAGObservation]]:
        state = DAGState.init(self.num_nodes, key)
        timestep = types.restart(
            observation=DAGObservation.from_state(state),
            extras={'is_valid_action': jnp.array(True)}
        )
        return (state, timestep)

    def step(self, state: DAGState, action: chex.Array) -> Tuple[DAGState, types.TimeStep[DAGObservation]]:
        source, target = jnp.divmod(action, self.num_nodes)

        def _step(state, source, target):
            is_valid_action = ~jnp.logical_or(
                state.adjacency[source, target],
                state.closure_T[source, target]
            )
            reward = (
                self.scorer.delta_score(state.adjacency, source, target)
                + self.prior.delta_score(state.adjacency, source, target)
            )
            state = state.update(source, target)
            timestep = types.transition(
                reward=reward,
                observation=DAGObservation.from_state(state),
                extras={'is_valid_action': is_valid_action}
            )
            return (state, timestep)

        def _reset(state, source, target):
            state = DAGState.init(self.num_nodes, state.key)
            timestep = types.termination(
                reward=jnp.zeros(()),
                observation=DAGObservation.from_state(state),
                extras={'is_valid_action': jnp.array(True)}
            )
            return (state, timestep)

        return jax.lax.cond(
            source == self.num_nodes,
            _reset, _step,
            state, source, target
        )

    def observation_spec(self) -> specs.Spec[DAGObservation]:
        return specs.Spec(DAGObservation, 'observation',
            adjacency=specs.BoundedArray(
                (self.num_nodes, self.num_nodes),
                dtype=jnp.float32,
                minimum=0.,
                maximum=1.,
                name='adjacency'
            ),
            mask=specs.BoundedArray(
                (self.num_nodes, self.num_nodes),
                dtype=jnp.float32,
                minimum=0.,
                maximum=1.,
                name='mask'
            )
        )

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(self.num_nodes ** 2 + 1, name='action')
