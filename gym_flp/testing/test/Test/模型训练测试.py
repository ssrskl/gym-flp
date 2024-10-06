# 模型训练测试
import numpy as np
from gym_flp.envs.FBS import FbsEnv
from gym_flp.util import FBSUtils
from stable_baselines3 import DQN


instance = "O7-maoyan"
env = FbsEnv(mode="human", instance=instance)
print(env.observation_space.shape)
env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
print(obs.shape)
# model = DQN("MlpPolicy", env, verbose=1, buffer_size=100000, batch_size=64)
# model.learn(total_timesteps=50000, reset_num_timesteps=False)

zero_array = np.zeros(7)
print(zero_array.shape)
