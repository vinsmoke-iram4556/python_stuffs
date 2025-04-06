import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from tabulate import tabulate  # For pretty-printing tables

# Define ODEs
def f1(t, y):
    return t * np.exp(3*t) - 2*y

def f1_odeint(y, t):
    return t * np.exp(3*t) - 2*y

def f2(t, y):
    return 1 + (t - y)**2

def f2_odeint(y, t):
    return 1 + (t - y)**2

# Exact solutions
def e1(t):
    return (1/3) * t * np.exp(3*t) - (1/25) * np.exp(3*t) + (1/25) * np.exp(-2*t)

def e2(t):
    return t + 1/(1-t)

# Problem i)
t1 = np.linspace(0, 1, 11)  # Reduced number of points for readable table
y0_1 = 0

# Using both methods
y1_ode = odeint(f1_odeint, y0_1, t1).flatten()
sol1 = solve_ivp(f1, [0, 1], [y0_1], t_eval=t1)
y1_ivp = sol1.y[0]

# Exact solution
y1_ex = e1(t1)

# Errors
err1_ode = np.abs(y1_ode - y1_ex)
err1_ivp = np.abs(y1_ivp - y1_ex)

# Problem ii)
t2 = np.linspace(2, 2.9, 10)  # Avoiding t=3 singularity and using fewer points
y0_2 = 1

# Using both methods
y2_ode = odeint(f2_odeint, y0_2, t2).flatten()
sol2 = solve_ivp(f2, [2, 2.9], [y0_2], t_eval=t2)
y2_ivp = sol2.y[0]

# Exact solution
y2_ex = e2(t2)

# Errors
err2_ode = np.abs(y2_ode - y2_ex)
err2_ivp = np.abs(y2_ivp - y2_ex)

# Create tables
def create_table(t, y_odeint, y_ivp, y_exact, err_odeint, err_ivp):
    table_data = []
    for i in range(len(t)):
        table_data.append([
            f"{t[i]:.2f}",
            f"{y_odeint[i]:.6e}", 
            f"{y_ivp[i]:.6e}", 
            f"{y_exact[i]:.6e}",
            f"{err_odeint[i]:.6e}", 
            f"{err_ivp[i]:.6e}"
        ])
    
    headers = ["t", "y_odeint", "y_solve_ivp", "y_exact", "Error_odeint", "Error_solve_ivp"]
    return tabulate(table_data, headers=headers, tablefmt="grid")

# Display tables
print("Table for Problem i): y' = te^(3t) - 2y; y(0) = 0")
print(create_table(t1, y1_ode, y1_ivp, y1_ex, err1_ode, err1_ivp))
print("\n")

print("Table for Problem ii): y' = 1 + (t-y)^2; y(2) = 1")
print(create_table(t2, y2_ode, y2_ivp, y2_ex, err2_ode, err2_ivp))

# Plotting (keeping the visualization as well)
fig, axs = plt.subplots(1,2, figsize=(12, 10))

# Problem i) solutions
axs[0].plot(t1, y1_ode, 'b-', label='odeint')
axs[0].plot(t1, y1_ivp, 'g--', label='solve_ivp')
axs[0].plot(t1, y1_ex, 'r-.', label='Exact')
axs[0].set_title("Problem i): y' = te^(3t) - 2y")
axs[0].set_xlabel('t')
axs[0].set_ylabel('y(t)')
axs[0].legend()
axs[0].grid(True)

# Problem ii) solutions
axs[1].plot(t2, y2_ode, 'b-', label='odeint')
axs[1].plot(t2, y2_ivp, 'g--', label='solve_ivp')
axs[1].plot(t2, y2_ex, 'r-.', label='Exact')
axs[1].set_title("Problem ii): y' = 1 + (t-y)^2")
axs[1].set_xlabel('t')
axs[1].set_ylabel('y(t)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()