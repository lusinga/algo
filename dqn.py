import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

#env = gym.make('Breakout-v0')
env = gym.make('SpaceInvaders-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("dqn_space_invaders")

#del model # remove to demonstrate saving and loading

#model = DQN.load("dqn_pendulum")

obs = env.reset()

score = 0

while True:
    print(score)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    score = score + 1

    if done:
        # obs = env.reset()
        print(score)
        break
