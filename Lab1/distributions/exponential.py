import numpy as np
import matplotlib.pyplot as plt

x = np.random.exponential(scale=1.0, size=100)
y = np.random.exponential(scale=1.0, size=100)
plt.scatter(x, y)
plt.title("Exponential Distribution")
plt.axis('equal')
plt.grid(True)
plt.show()
