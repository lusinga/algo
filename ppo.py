import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env

# Parallel environments
#env = make_vec_env('CartPole-v1', n_envs=4)
# env = gym.make('SpaceInvaders-v0')
env = gym.make('Pong-v0')

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
#model.save("ppo_pong")

#del model # remove to demonstrate saving and loading

#model = PPO.load("ppo_pong")

obs = env.reset()

score = 0
wins = 0

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    score = score + 1
    # print(dones)
    if rewards > 0:
        wins += rewards
        print('win!!!', rewards)

    if done:
        # obs = env.reset()
        print('finished', score)
        print('wins:', wins)
        break

