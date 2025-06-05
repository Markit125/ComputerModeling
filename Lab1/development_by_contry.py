import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
from numpy.random import uniform
from collections import defaultdict

def read_input(default_value, ask_line, value_type):
    parameter = input(ask_line)
    return default_value if parameter == "" else value_type(parameter)

df = pd.read_csv('dataset/country-data.csv')
df.set_index('country', inplace=True)

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df.values)

grid_size = read_input(10, "Enter grid size:\n", int)
iteration_count = read_input(1000, "Enter iteration count:\n", int)
learning_rate = read_input(0.5, "Enter learning rate:\n", float)
sigma = read_input(1, "Enter sigma:\n", int)


som_x, som_y = grid_size, grid_size
som = MiniSom(x=som_x, y=som_y, input_len=data_normalized.shape[1], sigma=sigma, learning_rate=learning_rate)
som.random_weights_init(data_normalized)
som.train_random(data_normalized, iteration_count)


bmu_groups = defaultdict(list)
for i, country in enumerate(df.index):
    bmu = som.winner(data_normalized[i])
    bmu_groups[bmu].append(country)

plt.figure(figsize=(10, 10))
plt.title("SOM U-Matrix")
plt.pcolor(som.distance_map().T, cmap='coolwarm')
plt.colorbar(label='Distance')

for (x, y), names in bmu_groups.items():
    for name in names:
        dx, dy = uniform(0, 0), uniform(-0.5, 0.4)
        plt.text(x + 0.5 + dx, y + 0.5 + dy, name[:16],
                 ha='center', va='center', fontsize=8, color='black')

plt.grid()
plt.show()
