import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC
from sb3_contrib import TRPO, TQC, ARS
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from CustomRewardWrapper import CustomRewardWrapper





os.environ["CUDA_VISIBLE_DEVICES"] = ""

algorithm_name = "PPO"

def train_model(env_id,n_envs,timesteps=200_000, tensorboard_log = f"./tqc_tensorboard/"
                , pretrained_model_path=None):
    env = make_vec_env(
    lambda: CustomRewardWrapper(gym.make("BipedalWalker-v3", hardcore=True)),
    n_envs=n_envs,
    vec_env_cls=SubprocVecEnv,
    )


    # Dicionário para selecionar o algoritmo
    algorithms = {
        "PPO": PPO,
        "A2C": A2C,
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
        print("Iniciando o treinamento com PPO...")
        model.learn(total_timesteps=timesteps)
        model.save(f"{algorithm_name}-BipedalWalker-v3-BothLegs-10M")
        print("Treinamento concluído e modelo salvo!")
    else:
        model = ModelClass("MlpPolicy", env, verbose=1,tensorboard_log=tensorboard_log)
        print("Iniciando o treinamento com o modelo...")
        model.learn(total_timesteps=timesteps)
        model.save(f"{algorithm_name}-BipedalWalker-v3-BothLegs-10M")
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
    n_envs = min(os.cpu_count(), 4)


    #env = gym.make("BipedalWalker-v3", n_envs=n_envs,render_mode=None, hardcore=True)

    #trained_model = train_model(env_id,n_envs=n_envs, timesteps=10_000_000,pretrained_model_path=None)

    trained_model = PPO.load("PPO-BipedalWalker-v3-original-1000000.zip")


    #trained_model.learn(total_timesteps=500_000, reset_num_timesteps=False, tb_log_name="tqc_hardcore_continued")
    #trained_model.save(f"{algorithm_name}-BipedalWalker-v3-original")

    test_env = CustomRewardWrapper(gym.make("BipedalWalker-v3", hardcore=True,render_mode="human"))



    #mean_reward, std_reward = evaluate_policy(trained_model, test_env, n_eval_episodes=10, render=False)
    #print(f"Recompensa Média: {mean_reward}, Desvio Padrão: {std_reward}")


    test_model(trained_model, test_env)