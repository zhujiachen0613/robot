# encoding:utf-8
# train.py
from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from time import sleep
import pybullet as p

from RL.myRL_env import RobotEnv


def callback(*params):
    info_dict = params[0]
    episode_rewards = info_dict['episode_rewards']
    print(f"episode total reward: {sum(episode_rewards)}")


def train():
    # 创建 RobotEnv 环境
    env = RobotEnv()

    # 包装环境，使其适用于 Stable Baselines3
    env = DummyVecEnv([lambda: env])

    # 初始化 DQN 模型，使用 MLP 策略网络
    model = DQN("MlpPolicy", env, verbose=1)

    # 训练模型
    print("开始训练，稍等片刻")
    model.learn(total_timesteps=10000, callback=callback)

    # 评估模型
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")

    # 保存训练好的模型
    model.save("dqn_robot_model")


def predict():
    env = RobotEnv(renders=True, discrete_actions=True)
    env = DummyVecEnv([lambda: env])
    model = DQN("MlpPolicy", env, verbose=1)
    model.load(load_path="./model", env=env)

    obs = env.reset()
    while True:
        sleep(1 / 60)
        action, state = model.predict(observation=obs)
        print(action)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
