import numpy as np
from scipy.optimize import linprog

# Coefficients of the cost function to minimize
c = [45, 40, 85, 65]

# Inequality constraints (A_ub * x <= b_ub)
# Convert >= constraints to <= by multiplying by -1
A_ub = [
    [-3, -4, -8, -6],   # Proteins: -3x1 -4x2 -8x3 -6x4 <= -800
    [-2, -2, -7, -5],   # Fat: -2x1 -2x2 -7x3 -5x4 <= -200
    [-6, -4, -7, -4]    # Carbohydrates: -6x1 -4x2 -7x3 -4x4 <= -700
]
b_ub = [-800, -200, -700]

# Variable bounds (non-negative)
bounds = [(0, None)] * 4

# Solve the linear program
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Display results
print("Optimal quantities (units):")
print(f"Food 1: {result.x[0]:.2f}, Food 2: {result.x[1]:.2f}, Food 3: {result.x[2]:.2f}, Food 4: {result.x[3]:.2f}")
print(f"\nMinimum cost: {result.fun:.2f} BDT")

