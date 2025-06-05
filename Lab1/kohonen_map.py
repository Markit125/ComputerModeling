import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.cluster import KMeans
import random

data = np.array([(random.random(), random.random()) for _ in range(100)])

som_size = 10
som = MiniSom(x=som_size, y=som_size, input_len=2, sigma=4, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, num_iteration=1000)

mapped = np.array([som.winner(d) for d in data])

print(som.winner(data[0]))

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(data)

plt.figure(figsize=(8, 8))
plt.title("Kohonen Map")
scatter = plt.scatter(mapped[:, 0], mapped[:, 1], c=labels, cmap='tab10', label='Data Points')
plt.grid(True)
plt.xlim(-1, som_size)
plt.ylim(-1, som_size)
plt.colorbar(scatter, ticks=range(n_clusters), label="Cluster")
plt.show()
