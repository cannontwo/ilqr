import numpy as np

def dynamics_d_state(theta, theta_dot):
    """
    Return the jacobian of the dynamics with respect to the state vector;
    f_x in the parlance of the iLQR equations.
    """
    return np.array([[1., 0.05],
                     [(30./2.) * np.cos(theta + np.pi) * 0.05, 1.]])

def dynamics_d_control():
    """
    Return the jacobian of the dynamics with respect to the control;
    f_u in the parlance of the iLQR equations.
    """
    return np.array([[0],
                     [(3. * 0.05)]])

def cost_grad_state(theta, theta_dot):
    """
    Return the gradient of the cost (negative reward) with respect to the state vector;
    l_x in the parlance of the iLQR equations.
    """
    return np.array([[2 * theta],
                     [0.2 * theta_dot]])

def cost_grad_control(u):
    """
    Return the gradient of the cost (negative reward) with respect to the control;
    l_u in the parlance of the iLQR equations.
    """
    return np.array([0.002 * u])    

def cost_hess_state_state():
    """
    Return an approximation of l_xx.
    """
    return np.array([[2, 0],
                     [0, 0.02]])

def cost_hess_control_state():
    """
    Return an approximation of l_ux.
    """
    return np.array([[0, 0]])

def cost_hess_control_control():
    """
    Return an approximation of l_uu.
    """
    return np.array([[0.000002]])

def Q_d_state(theta, theta_dot, v_x_prime):
    """
    Return Q_x.
    """
    return cost_grad_state(theta, theta_dot) + np.dot(dynamics_d_state(theta, theta_dot).T, v_x_prime)

def Q_d_control(u, v_x_prime):
    """
    Return Q_u.
    """
    return cost_grad_control(u) + np.dot(dynamics_d_control().T, v_x_prime)

def Q_d_state_d_state(theta, theta_dot, v_xx_prime):
    """
    Return Q_xx.
    """
    return cost_hess_state_state() + np.dot(dynamics_d_state(theta, theta_dot).T, np.dot(v_xx_prime, dynamics_d_state(theta, theta_dot)))

def Q_d_control_d_state(theta, theta_dot, v_xx_prime):
    """
    Return Q_ux.
    """
    return cost_hess_control_state() + np.dot(dynamics_d_control().T, np.dot(v_xx_prime, dynamics_d_state(theta, theta_dot)))

def Q_d_control_d_control(theta, theta_dot, v_xx_prime):
    """
    Return Q_uu.
    """
    return cost_hess_control_control() + np.dot(dynamics_d_control().T, np.dot(v_xx_prime, dynamics_d_control()))
