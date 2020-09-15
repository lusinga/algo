import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env

# Parallel environments
#env = make_vec_env('CartPole-v1', n_envs=4)
env = gym.make('SpaceInvaders-v0')

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
#model.save("ppo_cartpole")

#del model # remove to demonstrate saving and loading

#model = PPO.load("ppo_cartpole")

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

