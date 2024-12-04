import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from CustomRewardWrapper import CustomRewardWrapper


def train_model_with_tqc(env, timesteps=200_000, tensorboard_log="./tqc_tensorboard/", pretrained_model_path=None):
    if pretrained_model_path:
        model = TQC.load(pretrained_model_path, env=env)
    else:
        model = TQC(
            policy='MlpPolicy',
            env=env,
            learning_rate=1e-4,
            batch_size=128,
            gamma=0.98,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    print("Iniciando o treinamento com TQC...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save("tqc-BipedalWalker-v3-hardcore2")
    print("Treinamento concluído e modelo salvo!")

    return model


def evaluate_model(model, env, n_eval_episodes=5):
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    print(f"Recompensa média: {mean_reward} ± {std_reward}")

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
    env = CustomRewardWrapper(gym.make("BipedalWalker-v3", render_mode=None, hardcore=True))

    #trained_model = train_model_with_tqc(env, timesteps=350_000)
    trained_model = TQC.load("tqc-BipedalWalker-v3-hardcore2")
    trained_model.set_env(env)

    trained_model.learn(total_timesteps=100_000)
    trained_model.save("tqc-BipedalWalker-v3-hardcore2")

    test_env = CustomRewardWrapper(gym.make("BipedalWalker-v3", render_mode="human", hardcore=True))

    #evaluate_model(trained_model, env, n_eval_episodes=10)

    test_model(trained_model, test_env)