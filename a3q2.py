import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the model equations
def model(t, z):
    x, y = z
    dx = -0.1 * x + 0.02 * x * y
    dy = 0.2 * y - 0.025 * x * y
    return [dx, dy]

# Initial conditions
x0, y0 = 6, 6  # initial populations in thousands


# Solve with more dense output points for accuracy
sol = solve_ivp(model, (0, 50),[x0,y0], method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)

# Get a much denser time grid for accurate crossing detection
dense_t = np.linspace(0, 50, 50000)
dense_z = sol.sol(dense_t)
dense_x = dense_z[0]  # dense predator values
dense_y = dense_z[1]  # dense prey values

# Find where populations are equal (after t=0)
equal_times = []
for i in range(1, len(dense_t)):
    # Skip the initial point where they're already equal
    if dense_t[i] < 0.1:
        continue    
    # Check if the difference crosses zero
    diff_curr = dense_x[i] - dense_y[i]
    diff_prev = dense_x[i-1] - dense_y[i-1]
    
    if diff_curr * diff_prev <= 0:  # Sign change indicates crossing
        t_cross = dense_t[i] if abs(diff_curr) < abs(diff_prev) else dense_t[i-1]
        equal_times.append(t_cross)

# Create and save the figure
plt.figure(figsize=(12, 8))
plt.plot(dense_t, dense_x, 'r-', linewidth=2, label='Predators')
plt.plot(dense_t, dense_y, 'b-', linewidth=2, label='Prey')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Population (thousands)', fontsize=12)
plt.title('Lotka-Volterra Model', fontsize=14)
plt.grid(True)

# Mark where populations are equal
if equal_times:
    first_equal = equal_times[0]
    pop_at_equal = sol.sol(first_equal)
    plt.plot(first_equal, pop_at_equal[0], 'go', markersize=10, 
             label=f'Equal at t={first_equal:.2f}')
    # Mark with a vertical line for clarity
    plt.axvline(x=first_equal, color='g', linestyle='--', alpha=0.5)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Display numerical results
print("\nResults summary:")
if equal_times:
    print(f"The two populations first become equal at time t ≈ {equal_times[0]:.4f}")
    population_value = sol.sol(equal_times[0])[0]
    print(f"At this time, both populations equal {population_value:.4f} thousand individuals")
else:
    print("No time found where populations are equal after t=0")
