import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

T = 40
r = 0.05
beta = 0.95
sigma_y = 0.05263
y_mean = np.exp(sigma_y**2 / 2)
y_std = np.sqrt(np.exp(2 * sigma_y**2) - np.exp(sigma_y**2))

c_t = np.zeros(T)
a_t = np.zeros(T + 1)
mean_c = np.zeros(T)
var_c = np.zeros(T)
mean_a = np.zeros(T)
var_a = np.zeros(T)
mean_y = np.zeros(T)
var_y = np.zeros(T)

np.random.seed(0) 
y_t = np.random.lognormal(0, sigma_y, T)

a_t[0] = 0
c_t[-1] = y_t[-1] - a_t[-1] 

for t in reversed(range(T - 1)):
    c_t[t] = c_t[t + 1] / (0.95 * (1 + r))

# Forward iteration for assets
for t in range(T):
    a_t[t + 1] = (1 + r) * a_t[t] + y_t[t] - c_t[t]
    
    mean_c[t] = np.mean(c_t[:t+1])
    var_c[t] = np.var(c_t[:t+1])
    mean_a[t] = np.mean(a_t[:t+1])
    var_a[t] = np.var(a_t[:t+1])
    mean_y[t] = np.mean(y_t[:t+1])
    var_y[t] = np.var(y_t[:t+1])

while abs(a_t[-1]) > 1e-6:
    c_t[-1] += a_t[-1] / (T * (1 + r) ** (T - 1))
    for t in reversed(range(T - 1)):
        c_t[t] = c_t[t + 1] / (0.95 * (1 + r))
    for t in range(T):
        a_t[t + 1] = (1 + r) * a_t[t] + y_t[t] - c_t[t]

    mean_c[-1] = np.mean(c_t)
    var_c[-1] = np.var(c_t)
    mean_a[-1] = np.mean(a_t)
    var_a[-1] = np.var(a_t)
    mean_y[-1] = np.mean(y_t)
    var_y[-1] = np.var(y_t)

fig, axs = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)

axs[0, 0].plot(range(1, T + 1), mean_y, label='Mean of Income $y_t$', color='blue')
axs[0, 0].set_title('Mean of Income $y_t$', pad=20)
axs[0, 0].set_xlabel('Time', labelpad=15)
axs[0, 0].set_ylabel('Mean', labelpad=15)

axs[0, 1].plot(range(1, T + 1), mean_c, label='Mean of Consumption $c_t$', color='green')
axs[0, 1].set_title('Mean of Consumption $c_t$', pad=20)
axs[0, 1].set_xlabel('Time', labelpad=15)
axs[0, 1].set_ylabel('Mean', labelpad=15)

axs[1, 0].plot(range(1, T + 1), mean_a, label='Mean of Assets $a_t$', color='red')
axs[1, 0].set_title('Mean of Assets $a_t$', pad=20)
axs[1, 0].set_xlabel('Time', labelpad=15)
axs[1, 0].set_ylabel('Mean', labelpad=15)

axs[1, 1].plot(range(1, T + 1), var_y, label='Variance of Income $y_t$', color='blue')
axs[1, 1].set_title('Variance of Income $y_t$', pad=20)
axs[1, 1].set_xlabel('Time', labelpad=15)
axs[1, 1].set_ylabel('Variance', labelpad=15)

axs[2, 0].plot(range(1, T + 1), var_c, label='Variance of Consumption $c_t$', color='green')
axs[2, 0].set_title('Variance of Consumption $c_t$', pad=20)
axs[2, 0].set_xlabel('Time', labelpad=15)
axs[2, 0].set_ylabel('Variance', labelpad=15)

axs[2, 1].plot(range(1, T + 1), var_a, label='Variance of Assets $a_t$', color='red')
axs[2, 1].set_title('Variance of Assets $a_t$', pad=20)
axs[2, 1].set_xlabel('Time', labelpad=15)
axs[2, 1].set_ylabel('Variance', labelpad=15)

plt.show()

end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.2f} seconds")
