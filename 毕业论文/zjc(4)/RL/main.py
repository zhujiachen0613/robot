# encoding:utf-8
# train.py
import time
from pybullet_envs.bullet import CartPoleBulletEnv
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from time import sleep
import pybullet as p

from myRL_env import DiscretizedActionWrapper, RobotEnv
import csv

# def callback(*params):
#     info_dict = params[0]
#     episode_rewards = info_dict["episode_rewards"]
#     print(f"episode total reward: {sum(episode_rewards)}")

from stable_baselines3.common.callbacks import BaseCallback


class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MyCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        在每个训练步骤之后调用。
        """
        # 打印当前的训练步数
        # time.sleep(0.1)
        print("Training step:", self.num_timesteps)
        return True  # 返回True表示继续训练，返回False表示停止训练


callback = MyCallback()


def train():
    # 创建 RobotEnv 环境
    env = RobotEnv()

    env = DiscretizedActionWrapper(env, bins=10)

    # 包装环境，使其适用于 Stable Baselines3
    env = DummyVecEnv([lambda: env])

    env.reset()

    print(env.action_space)

    # 初始化 DQN 模型，使用 MLP 策略网络
    # model = DQN("CnnPolicy", env, verbose=1)
    model = DQN("MlpPolicy", env, verbose=1,gamma=0.9)
    # model = DDPG("MlpPolicy", env, verbose=1, gamma=0.9)

    # 训练模型
    print("开始训练，稍等片刻")
    model.learn(total_timesteps=25000, callback=callback)

    # 评估模型
    # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)
    # print(f"Mean reward: {mean_reward}")

    # 保存训练好的模型
    model.save("dqn_3_50000")
    env.close()


def predict():
    env = RobotEnv()
    env = DiscretizedActionWrapper(env, bins=10)
    env = DummyVecEnv([lambda: env])
    model = DQN("MlpPolicy", env, verbose=1, gamma=0.8)
    # model = DDPG("MlpPolicy", env, verbose=1, gamma=0.9)
    model.load(path="dqn_model_1_r1+r2.zip", env=env)
    data = []

    obs = env.reset()

    i = 0
    while True:
        sleep(0.1)
        i += 1
        action, state = model.predict(observation=obs)
        print(action)
        obs, reward, done, info = env.step(action)
        data.append([action, state, obs, reward, done, info])
        if done:
            break
        if i > 40:
            break

    print(data)
    with open("example.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    env.close()


if __name__ == "__main__":
    train()

    #predict()
