import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.ppo.ppo import PPO

instance = "O7-maoyan"

# 创建环境和模型
env = gym.make("fbs-v0", instance=instance, mode="human")
# model = DQN("MlpPolicy", env, verbose=1,tensorboard_log="logs")
model = DQN("MlpPolicy", env, verbose=1)


# 初始化模型（这会确保所有组件都被正确设置）
model.learn(total_timesteps=1, reset_num_timesteps=False)

total_episodes = 1000
max_steps_per_episode = 10000

for episode in range(total_episodes):
    obs = env.reset()
    for step in range(max_steps_per_episode):
        action, _states = model.predict(obs, deterministic=False)
        action_number = action.item()
        action = action
        new_obs, reward, done, info = env.step(action_number)
        # 存储transition
        model.replay_buffer.add(obs, new_obs, action, reward, done, [info])
        obs = new_obs
        if done:
            break

    # 每个episode结束后进行训练
    if model.replay_buffer.size() > model.batch_size:
        model.train(gradient_steps=step, batch_size=model.batch_size)

    print(f"Episode {episode+1}/{total_episodes}, Reward: {reward}, Steps: {step+1}")

    # 每100个episodes保存一次模型
    if (episode + 1) % 100 == 0:
        model.save(f"dqn-fbs-episode-{episode+1}-{instance}")

# 训练结束后保存最终模型
model.save("dqn_fbs_final")
