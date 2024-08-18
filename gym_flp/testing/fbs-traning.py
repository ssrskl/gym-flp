import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.ppo.ppo import PPO

# 创建环境和模型
env = gym.make("fbs-v0", instance="MB12", mode="human")
# model = DQN("MlpPolicy", env, verbose=1,tensorboard_log="logs")
model = DQN("MlpPolicy", env, verbose=1)
# model = DQN(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0001,
#     buffer_size=200000,
#     learning_starts=50000,
#     batch_size=32,
#     tau=1.0,
#     gamma=0.99,
#     train_freq=4,
#     gradient_steps=1,
#     replay_buffer_class=None,
#     replay_buffer_kwargs=None,
#     optimize_memory_usage=False,
#     target_update_interval=10000,
#     exploration_fraction=0.9,
#     exploration_initial_eps=1.0,
#     exploration_final_eps=0.05,
#     max_grad_norm=10,
#     policy_kwargs=None,
#     verbose=1,
#     seed=42,
#     device="auto",
#     _init_setup_model=True,
# )

# 训练模型
# model.learn(total_timesteps=4000 * 10)
# model.save("dqn-fbs-40000-MB12")

# for round in range(100):
#     print("现在训练轮数:", round)
#     model.learn(total_timesteps=4000)


for episode in range(100):
    obs = env.reset()
    done = False
    total_timesteps = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        action = action.item()
        obs, reward, done, info = env.step(action)
        model.learn(total_timesteps=1, reset_num_timesteps=False)  # 每步更新模型
        total_timesteps += 1
        if total_timesteps >= 10000:
            break
    print("Episode: {}, Total Timesteps: {}".format(episode, total_timesteps))

model.save("dqn-fbs-40000-MB12")
