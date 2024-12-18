from gymnasium import RewardWrapper
import numpy as np

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # 1. Incentiva o movimento para frente
        forward_velocity = obs[2]  
        reward += 0.3 * np.clip(forward_velocity, 0, 1.0)

        # 2. Penaliza o agente por ficar muito tempo parado (baixo movimento)
        if abs(forward_velocity) < 0.1:
            reward -= 0.2  

        # 3. Penalidade alta ao encerrar o episódio por queda
        if done:
            reward -= 25.0 

        # 4. Penaliza a inclinação do torso para manter estabilidade
        torso_angle = abs(obs[0])
        reward -= 0.3 * torso_angle

        #Obriga a andar com as duas pernas
        left_force = obs[8]
        right_force = obs[13]
        force_diff = abs(left_force - right_force)
        reward -= 0.1 * force_diff

        return obs, reward, done, truncated, info
