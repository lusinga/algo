import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn import CnnPolicy

# env = gym.make('Breakout-v0')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('Adventure-v4')
# env = gym.make('AirRaid-v0')
# env = gym.make('AirRaid-v0')
# env = gym.make('SpaceInvaders-v4')
# env = gym.make('Alien-v0')
# env = gym.make('Amidar-v0')
# env = gym.make('Assault-v0')
# env = gym.make('Asterix-v0')
# env = gym.make('Asteroids-v0')
# env = gym.make('Atlantis-v0')
# env = gym.make('BankHeist-v0') # 10000
# env = gym.make('BattleZone-v0')
# env = gym.make('BeamRider-v0') # 10000 主动前进型的
# env = gym.make('Berzerk-v0')
# env = gym.make('Bowling-v0')
# env = gym.make('Boxing-v0')
# env = gym.make('Carnival-v0')
# env = gym.make('Centipede-v0')
env = gym.make('Pong-v0')

save_file = 'dqn_pong';

print(env.action_space)
print(env.get_action_meanings())

model = DQN(CnnPolicy, env, verbose=1)
#model = DQN.load(save_file)
model.set_env(env)
# model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save(save_file)

obs = env.reset()

score = 0
rewards_sum = 0

while True:
    # print(score)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    score = score + 1
    # rewards_sum += reward
    if reward > 0:
        print('win!!!', reward)

    if done:
        # obs = env.reset()
        print('finished', score)
        break
