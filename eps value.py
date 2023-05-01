from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('Measurements2.csv', parse_dates=['timestamp'], infer_datetime_format=True, usecols=['timestamp', 'flow'])

# Compute the k-distance graph
k = 10  # the value of k for k-nearest neighbors
nbrs = NearestNeighbors(n_neighbors=k+1).fit(data[['flow']])
distances, indices = nbrs.kneighbors(data[['flow']])
k_distance = np.sort(distances[:, -1])

# Plot the k-distance graph
plt.plot(np.arange(len(k_distance)), k_distance)
plt.xlabel('Row index')
plt.ylabel(f'{k}-distance')
plt.title(f'{k}-distance graph')
plt.show()
