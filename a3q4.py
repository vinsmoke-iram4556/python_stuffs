import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0  # Length of pendulum in feet
g = 32.17  # Acceleration due to gravity in ft/s^2

# Initial conditions
theta_0 = np.pi/6  # Initial angle (radians)
omega_0 = 0.0  # Initial angular velocity (radians/s)

# Time parameters
t_0 = 0
t_f = 2
dt = 0.01
n_steps = int((t_f - t_0) / dt) + 1

# Define the system of first-order ODEs
def pendulum_system(t, y):
    """
    y[0] = θ (angle)
    y[1] = ω (angular velocity)
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

# RK4 method for system of ODEs
def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Arrays to store results
t_values = np.linspace(t_0, t_f, n_steps)
theta_values = np.zeros(n_steps)
omega_values = np.zeros(n_steps)

# Initial state
y = np.array([theta_0, omega_0])

# Solve using RK4
for i in range(n_steps):
    t = t_0 + i*dt
    theta_values[i] = y[0]
    omega_values[i] = y[1]
    if i < n_steps - 1:  # Don't step beyond the final time
        y = rk4_step(pendulum_system, t, y, dt)

# Print the results in a table
print(f"{'t (s)':^10}{'θ (rad)':^15}{'θ (deg)':^15}")
print("-" * 40)
for i in range(n_steps):
    print(f"{t_values[i]:10.2f}{theta_values[i]:15.6f}{np.degrees(theta_values[i]):15.6f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, theta_values, 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Angle θ (radians)')
plt.title('Pendulum Motion: θ vs Time')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.show()