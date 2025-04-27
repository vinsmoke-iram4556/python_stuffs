import numpy as np
from scipy.optimize import linprog
from fractions import Fraction

def solve_game(payoff_matrix):
    """
    Solve a two-player zero-sum game using linear programming.
    
    Args:
        payoff_matrix: The payoff matrix where rows are Player A's strategies
                      and columns are Player B's strategies.
    
    Returns:
        The value of the game and optimal strategies for both players.
    """
    # Step 1: Check if we need to shift the payoff matrix to make it positive
    min_value = np.min(payoff_matrix)
    if min_value < 0:
        shift = -min_value
        shifted_payoff = payoff_matrix + shift
    else:
        shift = 0
        shifted_payoff = payoff_matrix.copy()
    
    m, n = shifted_payoff.shape
    
    # Step 2: Solve for player A's strategy
    # For player A's problem, we need to convert to standard form:
    # Maximize z = x_1 + x_2 + ... + x_m
    # Subject to: 
    #   a_11 * x_1 + a_21 * x_2 + ... + a_m1 * x_m <= 1
    #   a_12 * x_1 + a_22 * x_2 + ... + a_m2 * x_m <= 1
    #   ...
    #   a_1n * x_1 + a_2n * x_2 + ... + a_mn * x_m <= 1
    #   x_1, x_2, ..., x_m >= 0
    
    # Objective: maximize sum(x)
    c_A = -np.ones(m)
    
    # Constraints: A^T * x <= 1
    A_ub = shifted_payoff.T
    b_ub = np.ones(n)
    
    # Solve the linear program
    res_A = linprog(c_A, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    # Player A's strategy is x / sum(x)
    x = res_A.x
    v_A = 1 / np.sum(x)
    p = x * v_A
    
    # Step 3: Solve for player B's strategy
    # For player B's problem, we need to convert to standard form:
    # Minimize z = y_1 + y_2 + ... + y_n
    # Subject to:
    #   a_11 * y_1 + a_12 * y_2 + ... + a_1n * y_n >= 1
    #   a_21 * y_1 + a_22 * y_2 + ... + a_2n * y_n >= 1
    #   ...
    #   a_m1 * y_1 + a_m2 * y_2 + ... + a_mn * y_n >= 1
    #   y_1, y_2, ..., y_n >= 0
    
    # Objective: minimize sum(y)
    c_B = np.ones(n)
    
    # Constraints: A * y >= 1
    A_ub_B = -shifted_payoff
    b_ub_B = -np.ones(m)
    
    # Solve the linear program
    res_B = linprog(c_B, A_ub=A_ub_B, b_ub=b_ub_B, method='highs')
    
    # Player B's strategy is y / sum(y)
    y = res_B.x
    v_B = 1 / np.sum(y)
    q = y * v_B
    
    # Step 4: Compute the value of the original game
    value = v_A - shift
    
    return value, p, q

# Define the game payoff matrix
# Values represent Player A's gains (and Player B's losses)
payoff_matrix = np.array([
    [-1, 7, 6],  # Player A strategy I vs Player B strategies I, II, III
    [-2, 5, 0],  # Player A strategy II vs Player B strategies I, II, III
    [8, -1, 12]  # Player A strategy III vs Player B strategies I, II, III
])

# Solve the game
value, strategy_A, strategy_B = solve_game(payoff_matrix)

# Print the results with higher precision
print(f"Value of the game: {value:.6f}")
print("\nOptimal strategy for Player A:")
for i, prob in enumerate(strategy_A):
    if prob > 1e-10:  # Only show non-zero probabilities
        print(f"Strategy {i+1}: {prob:.6f}")

print("\nOptimal strategy for Player B:")
for i, prob in enumerate(strategy_B):
    if prob > 1e-10:  # Only show non-zero probabilities
        print(f"Strategy {i+1}: {prob:.6f}")

# Verify the value
expected_payoff = strategy_A @ payoff_matrix @ strategy_B
print(f"\nVerified value: {expected_payoff:.6f}")

# Calculate the exact value as a fraction
try:
    exact_value = Fraction(float(value)).limit_denominator(1000)
    print(f"\nValue as a fraction: {exact_value}")
    print(f"Value as decimal: {float(exact_value):.10f}")
except:
    pass

# Try to express the value as 55/17 if it's close
if abs(value - 55/17) < 1e-10:
    print("\nThe value appears to be exactly 55/17 = 3.235294117647059")
