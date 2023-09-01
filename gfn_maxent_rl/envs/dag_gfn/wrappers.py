import numpy as np
import gym


class RewardCorrection(gym.Wrapper):
    def __init__(self, env, alpha=1.):
        super().__init__(env)
        self.alpha = alpha  # Temperature parameter

    def step(self, actions):
        num_edges = np.sum(self.env._state['adjacency'], dtype=np.float32)  # t
        observations, rewards, terminated, truncated, infos = self.env.step(actions)

        # Correct the reward by subtracting log(t + 1), where t is the number
        # of edges in the current graph
        correction = np.where(terminated | truncated, -np.log1p(num_edges), 0.)
        rewards = rewards + self.alpha * correction

        return (observations, rewards, terminated, truncated, infos)
