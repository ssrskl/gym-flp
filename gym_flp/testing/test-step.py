import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3 import DQN

env = gym.make("fbs-v0", instance="O8", mode="human")
obs = env.reset(flag=True)
env.render()
obs, rewards, dones, info = env.step(env.action_space.sample())
env.render()
env.close()
