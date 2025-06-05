import numpy as np
import matplotlib.pyplot as plt

x = np.random.beta(a=10, b=1, size=100)
y = np.random.beta(a=5, b=1, size=100)
plt.scatter(x, y)
plt.title("Beta Distribution (a=2,5 / b=5,1)")
plt.axis('equal')
plt.grid(True)
plt.show()
