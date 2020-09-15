import gym

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy, CnnPolicy
from stable_baselines3.common.cmd_util import make_vec_env

# Parallel environments
# env = make_vec_env('SpaceInvaders-v0', n_envs=4)
# env = gym.make('SpaceInvaders-v0')
env = gym.make('Pong-v0')

# model = A2C(MlpPolicy, env, verbose=1)
# model = A2C(CnnPolicy, env, verbose=1)
model = A2C.load("a2c_pong")
model.set_env(env)
model.learn(total_timesteps=10000)
model.save("a2c_pong")

#del model # remove to demonstrate saving and loading


obs = env.reset()

score = 0
wins = 0

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    score = score + 1
    if rewards > 0:
        wins += rewards
        print('win!!!', rewards)
    # print(info)
    # print(dones)

    if done:
        print('finished', score)
        print('wins:', wins)
        break
