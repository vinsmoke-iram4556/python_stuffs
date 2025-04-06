import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def odes(t, y):
    return [y[1], y[2], -2*y[4]**2 + y[3], y[4], y[5], -y[1]**3 + y[4] + y[0] + np.sin(t)]

y0 = [float(x) for x in input("Enter initial conditions (x1(0) x1'(0) x1''(0) x2(0) x2'(0) x2''(0)): ").split()]
sol = solve_ivp(odes, (0, 10), y0, t_eval=np.linspace(0, 10, 1000))
plt.subplot(2, 1, 1); plt.plot(sol.t, sol.y[0]); plt.ylabel('x1');plt.xlabel('t'); plt.grid(True)
plt.subplot(2, 1, 2); plt.plot(sol.t, sol.y[3]); plt.ylabel('x2'); plt.xlabel('t'); plt.grid(True)
plt.show()