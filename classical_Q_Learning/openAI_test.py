import gym
import matplotlib.pyplot as plt
from gym.utils import play

env = gym.make('Breakout-v0')


# play.play(env, zoom=3)
for i in range(20):
    env.render(mode='human')


# matplotlib inline
array = env.render(mode='rgb_array')
print(array.shape)
plt.imshow(array)

plt.show()

env.close()