from ilqr import *

run_inv_pend_ilqr(np.array([np.pi, 0.0]), 100, lamb_factor=1.1, alpha_factor=1.2, render=False)

#for i in range(100):
#    high = np.array([np.pi, 1])
#    run_inv_pend_ilqr(np.random.uniform(-high, high), 100, lamb_factor=1.1, alpha_factor=1., render=False)
