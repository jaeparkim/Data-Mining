import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


global epsilon


def l2_norm(x):
    return math.sqrt(np.sum(np.square(x)))


def cosine(y, z):
    return np.dot(y, z) / l2_norm(y) / l2_norm(z)


def three_points(centered_data, corr_matrix):
    non_neg_matrix = np.abs(corr_matrix)
    least_corr = np.unravel_index(np.argmin(non_neg_matrix), non_neg_matrix.shape)

    # making sure the correlation matrix is a square matrix
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        return
    for i in range(corr_matrix.shape[0]):
        corr_matrix[i][i] = 0

    most_corr = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    anti_corr = np.unravel_index(np.argmin(corr_matrix), corr_matrix.shape)

    plot_interesting_pairs(centered_data, most_corr, anti_corr, least_corr)


def plot_interesting_pairs(centered_data, most_corr, neg_corr, least_corr):
    plt.plot(centered_data[:, most_corr[0]], centered_data[:, most_corr[1]], 'o', color='black', markersize=1)
    plt.title("most positively correlated attributes")
    plt.xlabel('Attribute #' + str(most_corr[0] + 1))
    plt.ylabel('Attribute #' + str(most_corr[1] + 1))
    plt.savefig("most_corr.png")
    plt.clf()

    plt.plot(centered_data[:, neg_corr[0]], centered_data[:, neg_corr[1]], 'o', color='black', markersize=1)
    plt.title("most negatively correlated attributes")
    plt.xlabel('Attribute #' + str(neg_corr[0] + 1))
    plt.ylabel('Attribute #' + str(neg_corr[1] + 1))
    plt.savefig("neg_corr.png")
    plt.clf()

    plt.plot(centered_data[:, least_corr[0]], centered_data[:, least_corr[1]], 'o', color='black', markersize=1)
    plt.title("least correlated attributes")
    plt.xlabel('Attribute #' + str(least_corr[0] + 1))
    plt.ylabel('Attribute #' + str(least_corr[1] + 1))
    plt.savefig("least_corr.png")
    plt.clf()


def find_eigen_vec(corr_matrix, epsilon):
    x_vec = np.random.rand(corr_matrix.shape[0], 1) * 10
    x_prev = x_vec
    x_vec_max = 0
    x_prev_max = 0
    loop = True

    while loop:
        x_vec = np.matmul(corr_matrix, x_vec)
        x_vec_max = np.amax(np.abs(x_vec))
        x_vec = x_vec / x_vec_max

        if l2_norm(x_vec - x_prev) < epsilon:
            print("eigenvalue", round(x_vec_max, 3))
            printformat()
            print("eigenvector without normalization\n", x_vec)
            loop = False
        else:
            x_prev = x_vec
            x_prev_max = x_vec_max

    return x_vec  # eigenvector


def printformat():
    print("--------------------------------------------------------------------")
    # print("\n")


if __name__ == '__main__':

    filename = sys.argv[1]
    epsilon = float(sys.argv[2])

    """
    part 1
    """
    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data_matrix = pandas_dataframe.to_numpy()

    mean_vec = data_matrix.sum(axis=0) / data_matrix.shape[0]
    print("mean vector\n", np.round(mean_vec, decimals=3))
    printformat()

    total_var = sum(np.square(data_matrix - mean_vec).sum(axis=0) / data_matrix.shape[0])  # divide by n-1?
    print("total variance\n", np.round(total_var, decimals=3))
    printformat()

    centered_data = data_matrix - mean_vec

    inner_product_cov = np.dot(np.transpose(centered_data), centered_data) / centered_data.shape[0]
    print("inner product covariance\n", np.round(inner_product_cov, decimals=3))
    printformat()

    outer_product_cov = sum([np.outer(centered_data[i], centered_data[i]) for i in range(centered_data.shape[0])]) / \
                        centered_data.shape[0]
    print("outer product covariance\n", np.round(outer_product_cov, decimals=3))
    printformat()

    matrix = centered_data.T
    corr_matrix = np.array([[cosine(i, j) for j in matrix] for i in matrix])
    print("correlation matrix\n", np.round(corr_matrix, decimals=3))
    printformat()

    three_points(centered_data, corr_matrix)

    """
    part 2
    """
    eigenvector = find_eigen_vec(inner_product_cov, epsilon)
    eigenvector = eigenvector / l2_norm(eigenvector)

    projection_values = np.sum(np.square(centered_data - eigenvector.T), axis=1)

    y = np.linspace(0, 0, num=projection_values.shape[0])

    plt.plot(projection_values, y, 'o', color='black', markersize=1)
    plt.title("data projections")
    plt.xlabel('scalar projection')
    plt.savefig("data_projections.png")

    sys.exit()
