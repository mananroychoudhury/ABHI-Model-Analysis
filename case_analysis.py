import numpy as np
import time

start_time = time.time()

T = 40
beta = 0.95
r = 0.05  
y_mean = 1.0267

min_assets = [-40, -10, 0]

for a_min in min_assets:
    print(f"\nAnalyzing case with minimum asset level a = {a_min}")

    c_t = np.zeros(T)
    a_t = np.zeros(T + 1)
    c_t[-1] = 1  
    a_t[0] = 0

    for t in reversed(range(T - 1)):
        c_t[t] = c_t[t + 1] / (0.95 * (1 + r))

    for t in range(T):
        a_t[t + 1] = max((1 + r) * a_t[t] + y_mean - c_t[t], a_min)

    
    while abs(a_t[-1]) > 1e-6:  
        c_t[-1] += a_t[-1] / (40 * (1 + r) ** 39)
        for t in reversed(range(T - 1)):
            c_t[t] = c_t[t + 1] / (0.95 * (1 + r))
        for t in range(T):
            a_t[t + 1] = max((1 + r) * a_t[t] + y_mean - c_t[t], a_min)

    print("Optimal consumption path:", c_t)
    print("Final asset holdings:", a_t)

end_time = time.time() 
runtime = end_time - start_time
print(f"Runtime: {runtime:.2f} seconds")
