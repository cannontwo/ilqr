from ilqr import *

final_controls = run_inv_pend_ilqr(np.array([0.0, -1.0, 0.0]), 100, lamb_factor=1.1, alpha_factor=1.2, render=False)


env = gym.make('Pendulum-v0')

env.reset()
env.unwrapped.state = np.array([np.pi, 0.0])

for control in final_controls:
    _, _, _, _ = env.step(np.array([control]))
    env.render()

#for i in range(100):
#    high = np.array([0.0, -1.0, 1])
#    run_inv_pend_ilqr(np.random.uniform(-high, high), 100, lamb_factor=1.1, alpha_factor=1., render=False)
