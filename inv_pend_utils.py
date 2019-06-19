import numpy as np

def dynamics_d_state(x, y, theta_dot):
    """
    Return the jacobian of the dynamics with respect to the state vector;
    f_x in the parlance of the iLQR equations.
    """
    delta = 0.05
    return np.array([[(-y * np.cos(np.arctan2(y, x) + (delta * theta_dot)))/(x**2 + y**2), (x * np.cos(np.arctan2(y, x) + (delta * theta_dot)))/(x**2 + y**2), delta * np.cos(np.arctan2(y, x) + (delta * theta_dot))],
                     [ (y * np.sin(np.arctan2(y, x) + (delta * theta_dot)))/(x**2 + y**2), (-x * np.sin(np.arctan2(y, x) + (delta * theta_dot)))/(x**2 + y**2), -delta * np.sin(np.arctan2(y, x) + (delta * theta_dot))],
                     [(30.0 * delta) / 2, 0, 1]])

def dynamics_d_control():
    """
    Return the jacobian of the dynamics with respect to the control;
    f_u in the parlance of the iLQR equations.
    """
    return np.array([[0],
                     [0],
                     [(3. * 0.05)]])

def cost_grad_state(x, y, theta_dot):
    """
    Return the gradient of the cost (negative reward) with respect to the state vector;
    l_x in the parlance of the iLQR equations.
    """
    return np.array([[2 * np.arctan2(y, x) * (-y / (x**2 + y**2))],
                     [2 * np.arctan2(y, x) * (x / (x**2 + y**2))],
                     [0.2 * theta_dot]])

def cost_grad_control(u):
    """
    Return the gradient of the cost (negative reward) with respect to the control;
    l_u in the parlance of the iLQR equations.
    """
    return np.array([0.002 * u])    

# Helper functions for approximating cost hessians
def r_x(x, y):
    return np.array([[-y / (x**2 + y**2), x / (x**2 + y**2), 0.0],
                     [0.0, 0.0, 0.316],
                     [0.0, 0.0, 0.0]])


def r_u():
    return np.array([[0.0],
                     [0.0],
                     [0.0316]])

def cost_hess_state_state(x, y):
    """
    Return an approximation of l_xx.
    """
    return 2 * np.dot(np.transpose(r_x(x, y)), r_x(x, y))

def cost_hess_control_state(x, y):
    """
    Return an approximation of l_ux.
    """
    return 2 * np.dot(np.transpose(r_u()), r_x(x, y))

def cost_hess_control_control():
    """
    Return an approximation of l_uu.
    """
    return 2 * np.dot(np.transpose(r_u()), r_u())

def Q_d_state(x, y, theta_dot, v_x_prime):
    """
    Return Q_x.
    """
    return cost_grad_state(x, y, theta_dot) + np.dot(dynamics_d_state(x, y, theta_dot).T, v_x_prime)

def Q_d_control(u, v_x_prime):
    """
    Return Q_u.
    """
    return cost_grad_control(u) + np.dot(dynamics_d_control().T, v_x_prime)

def Q_d_state_d_state(x, y, theta_dot, v_xx_prime):
    """
    Return Q_xx.
    """
    return cost_hess_state_state(x, y) + np.dot(dynamics_d_state(x, y, theta_dot).T, np.dot(v_xx_prime, dynamics_d_state(x, y, theta_dot)))

def Q_d_control_d_state(x, y, theta_dot, v_xx_prime):
    """
    Return Q_ux.
    """
    return cost_hess_control_state(x, y) + np.dot(dynamics_d_control().T, np.dot(v_xx_prime, dynamics_d_state(x, y, theta_dot)))

def Q_d_control_d_control(x, y, theta_dot, v_xx_prime):
    """
    Return Q_uu.
    """
    return cost_hess_control_control() + np.dot(dynamics_d_control().T, np.dot(v_xx_prime, dynamics_d_control()))
