import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make('Pendulum-v0')
# env = gym.make('Breakout-v0')
env = gym.make('SpaceInvaders-v0')
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
# env = gym.make('Pong-v0')

print(env.action_space)
print(env.get_action_meanings())

# The noise objects for DDPG
n_actions = len(env.get_action_meanings())
print(n_actions)
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
model = DDPG('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
#model.save("ddpg_pendulum")
#env = model.get_env()

#del model # remove to demonstrate saving and loading

#model = DDPG.load("ddpg_pendulum")

score = 0

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

    score = score + 1
    print(dones)

    if dones.any():
        # obs = env.reset()
        print('finished', score)
        break
