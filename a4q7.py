import numpy as np
from scipy.optimize import linear_sum_assignment

# Create our profit matrix from the given table
# Rows represent salesmen (A, B, C, D)
# Columns represent cities (1, 2, 3, 4)
profit_matrix = np.array([
    [16, 10, 14, 11],  # Salesman A
    [14, 11, 15, 15],  # Salesman B
    [15, 15, 13, 12],  # Salesman C
    [13, 12, 14, 15]   # Salesman D
])

# Since the Hungarian algorithm minimizes cost, we negate the profit matrix
# to convert the maximization problem to a minimization one
cost_matrix = -profit_matrix

# Apply the Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Calculate the total profit
total_profit = profit_matrix[row_ind, col_ind].sum()

# Display the optimal assignment
salesmen = ['A', 'B', 'C', 'D']
cities = ['1', '2', '3', '4']

print("Optimal assignment:")
for i, j in zip(row_ind, col_ind):
    print(f"Salesman {salesmen[i]} should be assigned to City {cities[j]} with profit {profit_matrix[i, j]} BDT per day")

print(f"\nTotal maximum profit: {total_profit} BDT per day")

# Verify all assignments
assignment_matrix = np.zeros_like(profit_matrix, dtype=int)
for i, j in zip(row_ind, col_ind):
    assignment_matrix[i, j] = 1

print("\nAssignment matrix (1 indicates assignment):")
print(assignment_matrix)