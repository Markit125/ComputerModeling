import numpy as np
import matplotlib.pyplot as plt

x = np.random.poisson(lam=5, size=100)
y = np.random.poisson(lam=10, size=100)
plt.scatter(x, y)
plt.title("Poisson Distribution")
plt.axis('equal')
plt.grid(True)
plt.show()
