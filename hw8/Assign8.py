import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans


def spectral_clustering(training_data, k, spread, obj):
    data_size = len(training_data)

    # Gaussian Kernel
    norm = squareform(pdist(training_data, 'euclidean'))
    kernel = np.exp(-(norm ** 2)/(2*spread))

    degree_matrix = np.sum(kernel, axis=0) * np.identity(data_size)
    L = np.zeros((data_size, data_size))

    if obj == 'ratio':
        L = degree_matrix - kernel
    elif obj == 'asymmetric':
        L = np.dot(np.linalg.inv(degree_matrix), degree_matrix - kernel)
    elif obj == 'symmetric':
        degree_matrix = np.diag(np.sqrt(np.diag(degree_matrix)))
        degree_matrix = np.linalg.inv(degree_matrix)
        L = np.dot(np.dot(degree_matrix, degree_matrix - kernel), degree_matrix)
    
    _, eigh = np.linalg.eigh(L)
    eigh = eigh[:, 0: k]
    # eigh = np.fliplr(eigh)

    norms = np.linalg.norm(eigh, ord=2, axis=1)
    for i in range(len(norms)):
        eigh[i] = eigh[i, :] / norms[i]

    kmeans = KMeans(n_clusters = k).fit(eigh)

    return kmeans.labels_


def f_measure(clustering_labels, training_target, k):
    training_target = list(np.int_(training_target))

    count = 0

    y = np.zeros(k)
    y_hat = np.zeros(k)
    table = np.zeros((k, 4) , dtype=int)
    
    # accuracy table calculation
    for target_index, prediction_index in zip(training_target, clustering_labels):
        if target_index == prediction_index:
            count += 1
        table[prediction_index][target_index] += 1
        y[target_index] += 1
        y_hat[prediction_index] += 1

    columnwise_sum = np.sum(table, axis=0, dtype=int)
    rowwise_sum = np.sum(table, axis=1, dtype=int)

    precision_vals = []
    recall_vals = []
    for i, row in enumerate(table):
        precision_vals.append(max(row) / rowwise_sum[i])
        x = row.tolist().index(max(row))
        recall_vals.append(max(row) / columnwise_sum[x])

    F_vals = []
    for i in range(k):
        F_vals.append((2 * precision_vals[i] * recall_vals[i]) / (precision_vals[i] + recall_vals[i]))
    
    return np.mean(F_vals)


def num_of_each_cluster(clustering_labels, k):
    count_each_cluster_size = np.zeros(k)
    for l in range(k):
        count = np.count_nonzero(clustering_labels == l)
        count_each_cluster_size[l] = count

    return count_each_cluster_size


'''
Spectral Clustering
'''
if __name__ == '__main__':
    filename = sys.argv[1]
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    spread = float(sys.argv[4])
    obj = sys.argv[5] 
    """
    obj == 
    'ratio'     : ratio cut clustering objective
    'asymmetric': asymmetric Laplacian matrix for normalized cut clustering objective
    'symmetric' : symmetric Laplacian matrix for normalized cut clustering objective
    """

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data = pandas_dataframe.to_numpy()
 
    response = data[:, 0]
    data_matrix = data[:, 1:]

    for idx , val in enumerate(response):
        if val <= 40:
            response[idx] = 0
        elif 40 < val and val <= 60:
            response[idx] = 1
        elif 60 < val and val <= 100:
            response[idx] = 2
        elif 100 < val:
            response[idx] = 3

    sample_size = 1000
    cluster_sizes = []
    F_scores = []

    for i in range(100):
        sss = StratifiedShuffleSplit(n_splits=1, test_size= sample_size, train_size=sample_size)
        train_idx, test_idx = next(sss.split(data_matrix, response.T)) # test has the samples of size 1000

        training_data = [data_matrix[i] for i in train_idx]
        training_target = [response[j] for j in train_idx]

        clustering_labels = spectral_clustering(training_data, k, spread, obj)
        cluster_sizes.append(num_of_each_cluster(clustering_labels, k))

        F = f_measure(clustering_labels, training_target, k)
        F_scores.append(F)

    print("w/ spread: %d \n" % spread)
    print("F measure score: ", max(F_scores))
    val = F_scores.index(max(F_scores))
    for i in range(k):
        print("Cluster %d size: %d" % (i, cluster_sizes[val][i]))

    sys.exit()
