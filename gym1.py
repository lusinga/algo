import gym
# env = gym.make('Breakout-ram-v0')
env = gym.make('Adventure-ram-v0')
# env = gym.make('SpaceInvaders-ram-v0')

env.reset()
print(env.action_space)
print(env.get_action_meanings())

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
