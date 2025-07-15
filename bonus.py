import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Constants
MAX_K = 10

# Computes inertia values for k = 1 to MAX_K using KMeans clustering
def compute_inertia_values(X):
    k_values = np.arange(1, MAX_K + 1)
    inertia_values = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        model = KMeans(n_clusters=k, init="k-means++", random_state=0)
        model.fit(X)
        inertia_values[i] = model.inertia_

    return k_values, inertia_values

# Finds the elbow point by measuring the max distance from each point to the line connecting the first and last points
def find_elbow_point(k_values, inertia_values):
    x1, y1 = k_values[0], inertia_values[0]
    xk, yk = k_values[-1], inertia_values[-1]

    denominator = np.hypot(xk - x1, yk - y1)
    distances = np.abs((yk - y1) * k_values - (xk - x1) * inertia_values + xk * y1 - yk * x1) / denominator
    elbow_idx = np.argmax(distances)

    return k_values[elbow_idx], inertia_values[elbow_idx]

# Plots the inertia graph and marks the elbow point, then saves the figure as 'elbow.png'
def plot_and_save_graph(k_values, inertia_values, elbow_point):
    plt.plot(k_values, inertia_values, marker='o')
    plt.xticks(k_values)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.scatter(*elbow_point, color='green', s=100, zorder=5)
    plt.annotate('Elbow Point',
                 xy=(elbow_point[0], elbow_point[1]),
                 xytext=(elbow_point[0] + 1.5, elbow_point[1] + 30),
                 arrowprops=dict(facecolor='black', shrink=0.15))
    plt.savefig('elbow.png')
    
# Loads dataset, computes inertia values, finds the elbow, and saves the plot
def main():
    iris_ds = datasets.load_iris()
    X = iris_ds.data
    k_values, inertia_values = compute_inertia_values(X)
    elbow_point = find_elbow_point(k_values, inertia_values)
    plot_and_save_graph(k_values, inertia_values, elbow_point)


if __name__ == "__main__":
    main()