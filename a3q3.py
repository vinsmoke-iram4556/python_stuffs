import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Model equations
def model(t, z):
    x, y = z
    return [x*(2-0.4*x-0.3*y), y*(1-0.1*y-0.3*x)]

# Initial conditions
cases = [[1.5,3.5], [1,1], [2,7], [4.5,0.5]]
case_labels = ['a', 'b', 'c', 'd']

# Create figure with 2x2 subplots for time series
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.flatten()

# Create separate figure for phase plane
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Plot each case
for i, start in enumerate(cases):
    # Solve ODE
    sol = solve_ivp(model, (0,50), start, t_eval=np.linspace(0,50,200))
    x, y = sol.y
    
    # Time series plot
    axs[i].plot(sol.t, x, 'b-', sol.t, y, 'r-')
    axs[i].set_title(f'Case {case_labels[i]}: ({start[0]},{start[1]})')
    axs[i].grid(True)
    
    # Phase plane trajectories
    ax2.plot(x, y, label=f'Case {case_labels[i]}')
    ax2.plot(start[0], start[1], 'o')

# Add nullclines to phase plane
x = np.linspace(0, 5, 50)
ax2.plot(x, (2-0.4*x)/0.3, 'b--', label='x-nullcline')
ax2.plot(x, np.maximum(0, (1-0.3*x)/0.1), 'r--', label='y-nullcline')

# Add equilibrium points
eq_pts = [[0,0], [5,0], [0,10], [2,4]]
ax2.plot([p[0] for p in eq_pts], [p[1] for p in eq_pts], 'ko')
ax2.text(2, 4, '(2,4)', fontsize=12)
ax2.set_xlabel('x'), ax2.set_ylabel('y')
ax2.set_xlim(0, 6), ax2.set_ylim(0, 10)
ax2.grid(True)
ax2.legend()

# Print equilibrium analysis
print("Coexistence point (2,4) stability analysis:")
J = np.array([[-0.8, -0.6], [-1.2, -0.8]])  # Jacobian at (2,4)
eigs = np.linalg.eigvals(J)
print(f"Eigenvalues: {eigs}")
print(f"Type: {'Saddle point (unstable)' if eigs[0]*eigs[1] < 0 else 'Stable'}")

# Print long-term predictions
print("\nLong-term behavior:")
for i, start in enumerate(cases):
    sol = solve_ivp(model, (0,100), start)
    x_end, y_end = sol.y[0,-1], sol.y[1,-1]
    outcome = "x extinct" if x_end < 0.1 else "y extinct" if y_end < 0.1 else "coexist"
    print(f"Case {case_labels[i]}: Starting at {start} → {outcome}")

plt.tight_layout()
plt.show()