import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Linear Shooting method with Euler's method
def shooting(h):
    # Setup problem
    N = int(1 / h)
    x = np.linspace(0, 1, N + 1)
    y_exact = np.exp(-10 * x)  # Exact solution
    
    # Initialize two solution vectors for our two IVPs
    u = np.zeros((N + 1, 2))  # For u(0)=1, u'(0)=0
    v = np.zeros((N + 1, 2))  # For v(0)=0, v'(0)=1
    u[0] = [1, 0]
    v[0] = [0, 1]
    
    # Solve both IVPs with Euler's method
    for i in range(N):
        # Update both solutions using y'' = 100y
        u[i+1] = u[i] + h * np.array([u[i, 1], 100 * u[i, 0]])
        v[i+1] = v[i] + h * np.array([v[i, 1], 100 * v[i, 0]])
    
    # Combine solutions to satisfy right boundary condition
    w = (np.exp(-10) - u[N, 0]) / v[N, 0]
    y = u[:, 0] + w * v[:, 0]
    
    # Create table
    table = []
    for i in range(len(x)):
        table.append([
            f"{x[i]:.2f}",
            f"{y[i]:.8f}",
            f"{y_exact[i]:.8f}",
            f"{abs(y[i] - y_exact[i]):.8f}"
        ])
    
    return x, y, table


# Get results for both step sizes
results = {}
for h in [0.1, 0.05]:
    x, y, table = shooting(h)
    results[h] = {'x': x, 'y': y}
    
    print(f"\n--- Results for h = {h} ---")
    print(tabulate(table, ["x", "y (numerical)", "y (exact)", "Absolute Error"], tablefmt="grid"))

# Solve using solve_bvp
def ode(x, y): return np.vstack((y[1], 100 * y[0]))
def bc(ya, yb): return np.array([ya[0] - 1, yb[0] - np.exp(-10)])

x_init = np.linspace(0, 1, 11)
y_init = np.zeros((2, x_init.size))
y_init[0] = 1

sol = solve_bvp(ode, bc, x_init, y_init)
x_fine = np.linspace(0, 1, 100)
y_bvp = sol.sol(x_fine)[0]

# Plot all solutions
plt.figure(figsize=(10, 6))
plt.plot(x_fine, np.exp(-10 * x_fine), 'g-', label='Exact: $y=e^{-10x}$')
plt.plot(results[0.1]['x'], results[0.1]['y'], 'ro-', label='Shooting (h=0.1)')
plt.plot(results[0.05]['x'], results[0.05]['y'], 'bs-', label='Shooting (h=0.05)')
plt.plot(x_fine, y_bvp, 'k--', label='solve_bvp')

plt.title('BVP Solution Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

