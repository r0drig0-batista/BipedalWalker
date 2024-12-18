import gymnasium as gym
import numpy as np

class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env, alpha=0.9):
        """
        Wrapper que suaviza as ações aplicadas.
        
        Parâmetros:
        -----------
        env : gym.Env
            O ambiente a ser envolvido.
        alpha : float
            Fator de suavização entre 0 e 1.
            - Próximo de 1 significa menor suavização (a ação atual tem mais peso).
            - Próximo de 0 significa maior suavização (a ação anterior tem mais peso).
        """
        super().__init__(env)
        self.alpha = alpha
        self.prev_act = None

    def action(self, act):
        # Se não temos uma ação anterior, iniciamos com a ação atual
        if self.prev_act is None:
            self.prev_act = act

        # Combina a ação atual com a anterior
        smoothed_act = self.alpha * act + (1 - self.alpha) * self.prev_act
        self.prev_act = smoothed_act

        return smoothed_act
