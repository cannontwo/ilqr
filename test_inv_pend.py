import gym
import math
import numpy as np

num_eps = 10
num_steps = 100

env = gym.make('Pendulum-v0')

def get_state(s):
    """
    Given the OpenAI Gym pendulum state [cos(theta), sin(theta), \dot{theta}], return [theta, \dot{theta}].
    """
    return np.array([math.atan2(s[1], s[0]), s[2]])

for ep in range(num_eps):
    env.reset()
    env.unwrapped.state = np.array([np.pi, 0.0])
    print("Episode {}".format(ep))

    for step in range(num_steps):
        #))
        obs, _, _, _ = env.step(env.action_space.sample())
        print("\tState is {}".format(get_state(obs)))

        env.render()
