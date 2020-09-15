import gym

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env

# Parallel environments
# env = make_vec_env('SpaceInvaders-v0', n_envs=4)
env = gym.make('SpaceInvaders-v0')

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
#model.save("a2c_cartpole")

#del model # remove to demonstrate saving and loading

#model = A2C.load("a2c_cartpole")

obs = env.reset()

score = 0

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    score = score + 1
    # print(dones)

    if done:
        # obs = env.reset()
        print('finished', score)
        break
