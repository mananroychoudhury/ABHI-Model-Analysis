import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Start the timer
start_time = time.time()

# Parameters
T = 40  # Time periods
r = 0.05  # Interest rate
beta = 0.95
sigma_y = 0.05263  # Standard deviation of log-normal distribution

# Simulation settings
num_simulations = 100000  # Number of paths to simulate

# Initialize arrays
all_y = np.zeros((num_simulations, T))
all_c = np.zeros((num_simulations, T))
all_a = np.zeros((num_simulations, T + 1))

# Generate simulations
np.random.seed(0)
for sim in range(num_simulations):
    # Initialize variables
    a_t = np.zeros(T + 1)
    c_t = np.zeros(T)
    y_t = np.random.lognormal(0, sigma_y, T)
    
    # Set initial conditions
    a_t[0] = 0  # Initial asset level
    
    # Simulate paths
    for t in range(T):
        if t == T - 1:
            c_t[t] = y_t[t] - a_t[t]  # Final period consumption
        else:
            # Consumption path
            c_t[t] = y_t[t] - a_t[t]  # Assume consumption matches the budget constraint
        a_t[t + 1] = (1 + r) * a_t[t] + y_t[t] - c_t[t]
        
    # Store results
    all_y[sim] = y_t
    all_c[sim] = c_t
    all_a[sim] = a_t  # Store full assets array including the last period

# Create a line plot of the evolution of distributions over time
fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

# Income distribution
for t in range(0, T, 5):  # Plot every 5 periods
    sns.kdeplot(all_y[:, t], label=f'Time {t + 1}', ax=axs[0])

axs[0].set_title('Distribution of Income $y_t$ over Time')
axs[0].set_ylabel('Density')
axs[0].legend()

# Consumption distribution
for t in range(0, T, 5):  # Plot every 5 periods
    sns.kdeplot(all_c[:, t], label=f'Time {t + 1}', ax=axs[1])

axs[1].set_title('Distribution of Consumption $c_t$ over Time')
axs[1].set_ylabel('Density')
axs[1].legend()

# Assets distribution
for t in range(0, T, 5):  # Plot every 5 periods
    sns.kdeplot(all_a[:, t], label=f'Time {t + 1}', ax=axs[2])

axs[2].set_title('Distribution of Assets $a_t$ over Time')
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Density')
axs[2].legend()

# Stop the timer
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.2f} seconds")

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
