import numpy as np

from gfn_maxent_rl.envs.phylo_gfn.trees import Leaf


def get_total_mutations(tree):
    if isinstance(tree, Leaf):
        return 0
    return (tree.mutations + get_total_mutations(tree.left)
        + get_total_mutations(tree.right))


class ExponentialReward:
    def __init__(self, num_nodes, scale=1., C=0.):
        self.num_nodes = num_nodes
        self.scale = scale
        self.C = C

        self._offset = (C / scale) / num_nodes

    def delta_score(self, tree):
        return self._offset - tree.mutations / self.scale

    def log_reward(self, trees):
        log_rewards = np.zeros((len(trees),), dtype=np.float_)
        for i, tree in enumerate(trees):
            total_mutations = get_total_mutations(tree)
            log_rewards[i] = (self.C - total_mutations) / self.scale

        return log_rewards
