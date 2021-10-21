import sys
import math
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

global alpha

if __name__ == '__main__':
    filename = sys.argv[1]
    alpha = float(sys.argv[2])

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data_matrix = pandas_dataframe.to_numpy()

    """
    part 1
    """
    mean_vec = data_matrix.sum(axis=0) / data_matrix.shape[0]
    centered_data = data_matrix - mean_vec

    covariance_matrix = np.dot(np.transpose(centered_data), centered_data) / centered_data.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    eigenvalues = eigenvalues[::-1]
    eigenvectors = np.fliplr(eigenvectors)

    total_var = np.sum(eigenvalues)

    sum = 0
    index = 0
    for i in eigenvalues:
        sum += i
        index += 1
        if sum / total_var >= alpha:
            print("%d dimensions required to capture \u03B1=%0.3f" % (index, alpha))
            break

    # MSE = var - eigenvalue (book - 7.19)
    #print("MSE along 1st eigenvector: ", total_var - eigenvalues[0])
    #print("MSE along 2nd eigenvector: ", total_var - eigenvalues[1])
    print("MSE along 3rd eigenvector: ", total_var - eigenvalues[0]- eigenvalues[1] - eigenvalues[2])

    # projection on the first and second eigenvectors
    x = np.dot(centered_data, eigenvectors[:, 0].T)  # / 1
    y = np.dot(centered_data, eigenvectors[:, 1].T)  # / 1

    plt.plot(x, y, 'o', color='black', markersize=1)
    plt.title("data projections")
    plt.xlabel('First eigenvector projection')
    plt.ylabel('Second eigenvector projection')
    plt.savefig("data_projections.png")
    plt.clf()

    sys.exit()
