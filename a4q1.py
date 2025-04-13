import numpy as np
import matplotlib.pyplot as plt

# Define the constraints as equations
x1 = np.linspace(-3,5,400)

# Constraint lines
c1 = (10-x1) / 2          # x1 + 2x2 = 10
c2 = 6 - x1                 # x1 + x2 = 6
c3 = x1 - 2                 # x1 - x2 = 2 (feasible side: x2 <= x1 - 2)
c4 = (x1 - 1) / 2           # x1 - 2x2 = 1 (feasible side: x2 <= (x1 - 1)/2)

# Plot constraints
plt.figure(figsize=(10, 8))
plt.plot(x1, c1, label=r'$x_1 + 2x_2 \leq 10$', color='blue')
plt.plot(x1, c2, label=r'$x_1 + x_2 \leq 6$', color='green')
plt.plot(x1, c3, label=r'$x_1 - x_2 \leq 2$', color='orange')
plt.plot(x1, c4, label=r'$x_1 - 2x_2 \leq 1$', color='purple')

# Axes and limits
plt.axhline(0, color='k', linewidth=0.8)
plt.axvline(0, color='k', linewidth=0.8)
plt.xlim(-1, 5)
plt.ylim(-1, 5)

# Define vertices of the feasible region
vertices = np.array([
    [0, 5],    # Intersection of x1 + 2x2 = 10 and x2-axis
    [2, 4],    # Intersection of x1 + 2x2 = 10 and x1 + x2 = 6
    [4, 2],    # Intersection of x1 + x2 = 6 and x1 - x2 = 2
    [3, 1],    # Intersection of x1 - x2 = 2 and x1 - 2x2 = 1
    [1, 0],    # Intersection of x1 - 2x2 = 1 and x1-axis
    [0, 0]     # Origin
])

# Shade the feasible region
plt.fill(vertices[:, 0], vertices[:, 1], color='lightgreen', alpha=0.3, label='Feasible Region')

# Highlight the optimal solution
plt.plot(4, 2, 'rd', markersize=8, label='Optimal Point (4, 2)')
plt.annotate('(4, 2)', xy=(4, 2), xytext=(4.2, 2.2), color='red')

# Labels and title
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title('Graphical Solution of Linear Programming Problem', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()