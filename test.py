import numpy as np
import matplotlib.pyplot as plt

# Define the range of n (discrete time indices)
n = np.arange(-10, 11)

# Define the discrete-time signal x[n] = cos(0.2πn)
x = np.cos(0.2 * np.pi * n)

# Plotting the signal
plt.stem(n, x, use_line_collection=True)
plt.title("Discrete-Time Signal: x[n] = cos(0.2πn)")
plt.xlabel("n (discrete time index)")
plt.ylabel("x[n]")
plt.grid(True)
plt.show()