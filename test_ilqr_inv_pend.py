from ilqr import *

run_inv_pend_ilqr(np.array([np.pi, 0]), 100, lamb_factor=1.1, alpha_factor=1., render=True)
