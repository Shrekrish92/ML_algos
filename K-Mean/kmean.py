import numpy as np
import random as random
import matplotlib.pyplot as plt

def K_mean(data, k):
    centroids = initialize_centroid(data, k)
    while True:
        old_cen = centroids
        labels = get_labels(data, centroids)
        centroids = update_cen(data, labels, k)

        if should_stop(old_cen , centroids):
            break
    return labels, centroids

def initialize_centroid(data, k):
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')
    for pt in data:
        x_min = min(pt[0], x_min)
        x_max = max(pt[0], x_max)
        y_min = min(pt[1], y_min)
        y_max = max(pt[1], y_max)
    centroids=[]
    for i in range(k):
        centroids.append([random_sample(x_min, x_max), random_sample(y_min, y_max)])
    return centroids

def random_sample(low, high):
    return low + (high - low)*random.random()

def get_labels(data, centroids):
    labels = []
    for pt in data:
        min_dist = float('inf')
        label = None
        for i, centroid in enumerate(centroids):
            new_dist = get_dist(pt, centroid)
            if new_dist<min_dist:
                min_dist = new_dist
                label=i
        labels.append(label)
    return labels

def get_dist(pt1, pt2):
    return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5

def update_cen(pts, labels, k):
    new_cens=[[0,0]for i in range(k)]
    cnt = [0] * k
    for pt, label in zip(pts, labels):
        new_cens[label][0] += pt[0]
        new_cens[label][1] += pt[1]
        cnt[label] += 1
    for i, (x,y) in enumerate(new_cens):
        new_cens[i] = (x/cnt[i], y/cnt[i])
    return new_cens

def should_stop(old_cens, new_cens, thresh=1e-5):
    total_movement = 0
    for oldpt, newpt in zip(old_cens, new_cens):
        total_movement += get_dist(oldpt, newpt)
    return total_movement < thresh

def generate_random_data(num_points, x_min, x_max, y_min, y_max):
    data = []
    for _ in range(num_points):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        data.append([x, y])
    return data


def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # List of colors for different clusters

    # Plot data points
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], color=colors[labels[i] % len(colors)], alpha=0.6)

    # Plot centroids
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color='k', marker='x', s=200)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-means Clustering')
    plt.grid(True)
    plt.show()

num_points = 50
x_min, x_max = 0, 10
y_min, y_max = 0, 10

data = generate_random_data(num_points, x_min, x_max, y_min, y_max)
print(data)
labels, centroids= K_mean(data, 5)
print(labels)

plot_clusters(data, labels, centroids)