from gymnasium import RewardWrapper
import numpy as np

class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        left_leg = obs[6]  
        right_leg = obs[8]  

        # Penaliza ficar com as pernas muito abertas
        leg_balance_penalty = -abs(left_leg - right_leg)  
        reward += 0.1 * leg_balance_penalty  

        # Incentiva o movimento para frente
        forward_velocity = obs[2]  
        torso_angle = abs(obs[0]) 
        reward += 0.3 * np.clip(forward_velocity, 0, 1.0)
        if torso_angle > 0.3:  # Penaliza alta velocidade com inclinação
            reward -= 0.5

        # Penaliza o agente por ficar muito tempo parado (baixo movimento)
        if abs(forward_velocity) < 0.1:
            reward -= 0.2  


        # Penaliza quedas ou desequilíbrios 
        reward -= 0.3 * torso_angle  

        impact_force = abs(obs[10] - obs[11])  # Exemplo de cálculo
        if impact_force > 0.5:  # Limite arbitrário para identificar impacto forte
            reward -= 0.2 * impact_force

        return obs, reward, done, truncated, info


