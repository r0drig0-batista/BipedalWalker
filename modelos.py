import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC
from sb3_contrib import TRPO, TQC, ARS
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os


def train_model(env_id, n_envs,timesteps=200_000, tensorboard_log="./tqc_tensorboard/", pretrained_model_path=None):
    env = make_vec_env(
        lambda: gym.make(env_id, hardcore=True),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    algorithm_name = "DQN"

    # Dicionário para selecionar o algoritmo
    algorithms = {
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
        "SAC": SAC,
        "TRPO": TRPO,
        "TQC": TQC,
        "ARS": ARS,
    }

    ModelClass = algorithms.get(algorithm_name)
    if not ModelClass:
        raise ValueError(f"Algoritmo {algorithm_name} não suportado!")
    
    if pretrained_model_path:
        model = ModelClass.load(pretrained_model_path, env=env)
    else:
        model = ModelClass("MlpPolicy", env, verbose=1,tensorboard_log=tensorboard_log)


    print("Iniciando o treinamento com PPO...")
    model.learn(total_timesteps=timesteps)
    model.save(f"{algorithm_name}-BipedalWalker-v3-original")
    print("Treinamento concluído e modelo salvo!")

    return model


def test_model(model, env):
    

    obs, _ = env.reset()
    total_reward = 0

    print("Testando o modelo...")
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if done or truncated:
            print(f"Recompensa total no episódio: {total_reward}")
            break

    env.close()
    

if __name__ == "__main__":
    env_id = "BipedalWalker-v3"
    n_envs = os.cpu_count()

    #env = gym.make("BipedalWalker-v3", n_envs=n_envs,render_mode=None, hardcore=True)

    trained_model = train_model(env_id,n_envs=n_envs, timesteps=2_000_000,pretrained_model_path=None)

    #trained_model = TQC.load("tqc-BipedalWalker-v3-hardcore2")

    #trained_model.learn(total_timesteps=100_000)
    #trained_model.save("tqc-BipedalWalker-v3-hardcore2")

    test_env = gym.make(env_id, render_mode="human", hardcore=True)
    test_model(trained_model, test_env)