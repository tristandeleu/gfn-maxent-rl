import numpy as np
import multiprocessing as mp
import jax

from queue import Empty as EmptyException
from collections import defaultdict

from gfn_maxent_rl.utils.exhaustive import compute_cache, push_source_flow_to_terminating_states
from gfn_maxent_rl.utils.metrics import jensen_shannon_divergence, entropy


class AsyncEvaluator:
    def __init__(self, env, algorithm, run, ctx=None, target={}):
        self.env = env
        self.algorithm = algorithm
        self.run = None if run.disabled else run
        self.ctx = mp.get_context(ctx)
        self.target = target

        self._log_policy = jax.jit(algorithm.log_policy)

        self._queue = self.ctx.Queue()
        self._process = self.ctx.Process(
            target=AsyncEvaluator._compute_metrics,
            args=(self._queue, env, target, self.run),
            daemon=True
        )
        self._process.start()

    def enqueue(self, params, state, step, batch_size=256):
        # Compute the cache
        cache = compute_cache(
            self.env,
            self._log_policy,
            params,
            state,
            batch_size=batch_size
        )

        # Add the cache & step to the queue for processing
        self._queue.put((step, cache))

    def join(self):
        self._queue.put(None)
        self._process.join()

    @staticmethod
    def _compute_metrics(queue, env, target, run):
        terminate = False
        while not terminate:
            # Create the batch of caches
            steps, raw_caches = [], defaultdict(list)
            while True:
                try:
                    result = queue.get(block=True, timeout=1)

                    if result is None:
                        terminate = True
                        break

                    step, cache = result
                    steps.append(step)
                    for key, log_probs in cache.items():
                        raw_caches[key].append(log_probs)
                except EmptyException:
                    break

            # Process the batch
            if steps:
                caches = dict()
                for key, log_probs in raw_caches.items():
                    caches[key] = np.stack(log_probs, axis=0)

                # Apply the push-flow function
                mdp_state_graph = push_source_flow_to_terminating_states(
                    env.mdp_state_graph,
                    caches
                )

                # Compute the log-probabilities
                log_probs = [dict() for _ in steps]
                for state, is_terminating in mdp_state_graph.nodes(data='terminating', default=False):
                    if is_terminating:
                        for i, log_prob in enumerate(mdp_state_graph.nodes[state]['log_prob']):
                            log_probs[i][state] = log_prob

                # Compute the metrics
                metrics = dict()
                for step, distribution in zip(steps, log_probs):
                    metrics[step] = {
                        'jsd': jensen_shannon_divergence(distribution, target['log_probs']),
                        'entropy': entropy(distribution),
                    }

                # Send to Wandb
                if run is not None:
                    for step, metric in metrics.items():
                        run.log({
                            **{f'metrics/{key}': value for (key, value) in metric.items()},
                            'metrics/step': step
                        })
