import numpy as np
import gym


class RewardCorrection(gym.Wrapper):
    def __init__(self, env, alpha=1.):
        super().__init__(env)
        self.alpha = alpha  # Temperature parameter

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        num_parents = self.env.num_parents(observations)

        correction = np.where(terminated | truncated, 0, -np.log1p(num_parents))
        rewards = rewards + self.alpha * correction

        return (observations, rewards, terminated, truncated, infos)
