import numpy as np
import math

def near(x, y):
    if abs(x - y) < 0.1:
        return True

    return False

def mod_get_state(s, prev_theta=0.0):
    """
    Given the OpenAI Gym pendulum state [cos(theta), sin(theta), \dot{theta}], return [theta, \dot{theta}].
    """
    theta = math.atan2(s[1], s[0])
    if (near(prev_theta, np.pi) and theta < 0.0) or (prev_theta > np.pi and theta < 0.0):
        theta = theta + 2 * np.pi
    elif (near(prev_theta, -np.pi) and theta > 0.0) or (prev_theta < -np.pi and theta > 0.0):
        theta = theta - 2 * np.pi

    if theta > 2 * np.pi:
        theta = theta - 2 * np.pi

    if theta < -2 * np.pi:
        theta = theta + 2 * np.pi

    return np.array([theta, s[2]])

x = np.linspace(-3*np.pi, 3*np.pi,  200)
s = np.zeros((len(x), 3))
s[:, 0] = np.cos(x)
s[:, 1] = np.sin(x)

prev_theta = -np.pi
for i in range(len(x)):
    theta = mod_get_state(s[i], prev_theta=prev_theta)[0]

    unmod_theta = math.atan2(s[i, 1], s[i, 0])
    print("Unmod: {} (sin = {}, cos = {})".format(unmod_theta, math.sin(unmod_theta), math.cos(unmod_theta)))
    print("Mod: {} (sin = {}, cos = {})".format(theta, math.sin(theta), math.cos(theta)))

    prev_theta = theta
