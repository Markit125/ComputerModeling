import numpy as np
import matplotlib.pyplot as plt

data = np.random.uniform(-5, 5, (100, 2))
plt.scatter(data[:, 0], data[:, 1])
plt.title("Uniform Distribution")
plt.axis('equal')
plt.grid(True)
plt.show()
