import gym
import math
import numpy as np
import matplotlib.pyplot as plt

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

def forward_pass(env, start_state, controls, render=False):
    """
    Run the iLQR forward pass, using OpenAI Gym as simulator.
    """
    env.reset()
    env.unwrapped.state = np.array(start_state)
    cost = 0.0
    states = [start_state]

    for control in controls:
        #print("Applying control {}".format(control))
        obs, reward, _, _ = env.step(np.array([control]))
        
        if render:
            env.render()

        states.append(get_state(obs))
        cost += -reward

    return (cost, states)

def backward_pass(env, lamb, alpha, states, controls):
    """
    Run the iLQR backward pass, using perfect information about dynamics and
    cost derivatives.
    """
    V = np.zeros((len(states),))
    V_x = np.zeros((len(states), 2, 1))
    V_xx = np.zeros((len(states), 2, 2))
    k_vec = np.zeros((len(controls), 1))
    K_vec = np.zeros((len(controls), 1, 2))

    V[-1] = 0.0
    V_x[-1] = np.zeros((2, 1))
    V_xx[-1] = np.zeros((2, 2))

    for i in range(len(states) - 2, -1, -1):
        Q_uu = Q_d_control_d_control(states[i][0], states[i][1], V_xx[i+1])
        inv_Q_uu = invert_mat(Q_uu, lamb)
        k = -1. * np.dot(inv_Q_uu, Q_d_control(controls[i], V_x[i+1]))
        K = -1. * np.dot(inv_Q_uu, Q_d_control_d_state(states[i][0], states[i][1], V_xx[i+1]))

        k_vec[i] = k
        K_vec[i] = K

        V[i] = (-1./2.) * np.dot(k.T, np.dot(Q_uu, k))
        V_x[i] = Q_d_state(states[i][0], states[i][1], V_x[i+1]) - np.dot(K.T, np.dot(Q_uu, k))
        V_xx[i] = Q_d_state_d_state(states[i][0], states[i][1], V_xx[i+1]) - np.dot(K.T, np.dot(Q_uu, K))

    new_controls = compute_controls(env, alpha, states, controls, k_vec, K_vec)
    return new_controls

def compute_controls(env, alpha, states, controls, k_vec, K_vec):
    """
    Do forward pass to compute controls, then return computed controls.
    """
    # TODO : Combine with other forward pass to save time.
    env.reset()
    env.unwrapped.state = np.array(states[0])
    new_states = [states[0]]
    new_controls = np.zeros_like(controls)

    for i in range(len(controls)):
        new_controls[i] = controls[i] + (alpha * k_vec[i]) + np.dot(K_vec[i], new_states[i] - states[i])
        obs, _, _, _ = env.step(np.array([new_controls[i]]))
        new_states.append(get_state(obs))

    return new_controls

def run_inv_pend_ilqr(start_state, num_controls, max_iter=100, lamb_factor=1.5, alpha_factor=1.5, max_lamb=100, conv_thresh=0.0001, render=False):
    """
    Do the full iLQR algorithm on the inverted pendulum, then return the
    sequence of computed controls.
    """
    lamb = 1.0
    alpha = 1.0
    cost = -float('inf')
    controls = np.zeros((num_controls,))
    
    costs = []

    # TODO : Initialize V_x and V_xx

    env = gym.make('Pendulum-v0')

    cost, states = forward_pass(env, start_state, controls)
    costs.append(cost)

    for i in range(max_iter):
        print("\nOn iteration {}".format(i))
        print("\tCost is currently {}".format(cost))
        print("\tControls are currently {}".format(controls))
        print("\tLambda is currently {}".format(lamb))

        new_controls = backward_pass(env, lamb, alpha, states, controls)

        # Snippet that implements the Levenberg-Marquardt heuristic
        new_cost, states = forward_pass(env, start_state, new_controls, render)
        print("Old cost was {}, new cost is {}".format(cost, new_cost))
        if new_cost < cost:
          controls = new_controls
          lamb /= lamb_factor
          #alpha /= alpha_factor
         
          if (abs(new_cost - cost)/cost) < conv_thresh:
            print("Convergence metric is {}".format(abs(new_cost - cost)/cost))
            print("Exiting because convergence criterion hit")
            break

          cost = new_cost
          costs.append(cost)
        else:
          lamb *= lamb_factor
          #alpha *= alpha_factor
          if lamb > max_lamb:
            break

    plt.plot(costs)
    plt.show()
    return controls


