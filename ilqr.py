import gym
import math
import numpy as np

from inv_pend_utils import *

###########################
# Utility functions
###########################

def get_state(s):
    """
    Given the OpenAI Gym pendulum state [cos(theta), sin(theta), \dot{theta}], return [theta, \dot{theta}].
    """
    return np.array([math.atan2(s[1], s[0]), s[2]])

def invert_mat(mat, lamb):
    """
    Invert matrix with Tikhonov regularization given by lamb.
    Return inverted matrix.
    """
    evals, evecs = np.linalg.eig(mat)
    evals[evals < 0] = 0.0
    evals += lamb
    inv = np.dot(evecs, np.dot(np.diag(1.0/evals), evecs.T))

    return inv

###########################
# iLQR Functions
###########################

def forward_pass(env, start_state, controls):
    """
    Run the iLQR forward pass, using OpenAI Gym as simulator.
    """
    env.reset()
    env.unwrapped.state = np.array(start_state)
    cost = 0.0
    states = [start_state]

    for control in controls:
        print("Applying control {}".format(control))
        obs, reward, _, _ = env.step(np.array([control]))
        env.render()

        states.append(get_state(obs))
        cost += -reward

    return (cost, states)

def backward_pass():
    """
    Run the iLQR backward pass, using perfect information about dynamics and
    cost derivatives.
    """
    # TODO

def run_inv_pend_ilqr(start_state, num_controls, max_iter=100, lamb_factor=2, max_lamb=5, conv_thresh=0.01):
    """
    Do the full iLQR algorithm on the inverted pendulum, then return the
    sequence of computed controls.
    """
    lamb = 1.0
    cost = -float('inf')
    controls = np.zeros((num_controls,))
    env = gym.make('Pendulum-v0')

    for i in range(max_iter):
        print("\nOn iteration {}".format(i))
        print("\tCost is currently {}".format(cost))
        print("\tControls are currently {}".format(controls))
        print("\tLambda is currently {}".format(lamb))

        # Snippet that implements the Levenberg-Marquardt heuristic
        new_cost, states = forward_pass(env, start_state, controls)
        if new_cost < cost:
          lamb /= lamb_factor
         
          if (abs(new_cost - cost)/cost) < conv_thresh:
            break
        else:
          lamb *= lamb_factor
          if lamb > max_lamb:
            break

        cost = new_cost

