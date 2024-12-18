import gymnasium as gym
import numpy as np

class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env, noise_std=0.01):
        super().__init__(env)
        self.noise_std = noise_std

    def action(self, act):
        # Adiciona ruído gaussiano às ações
        noisy_act = act + np.random.normal(0, self.noise_std, size=act.shape)
        # Garante que as ações fiquem no intervalo [-1, 1]
        clipped_act = np.clip(noisy_act, -1, 1)
        return clipped_act
