import sys
import math
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def find_best_spread(data_matrix, binary_data, spread_val):
    # Gaussian Kernel
    x_i = data_matrix.shape[0]
    kernel_matrix = np.zeros((x_i, x_i))
    for i in range(x_i):
        for j in range(x_i):
            kernel_matrix[i, j] = l2_norm(data_matrix[i], data_matrix[j])
    kernel_matrix = np.exp((-1) * (kernel_matrix / (2 * spread_val)))

    # class-wise kernel matrices
    K_c1 = []
    K_c2 = []
    for i in range(0, binary_data.shape[0]):
        if binary_data[i] == 0:
            K_c1.append(kernel_matrix[:, i])
        elif binary_data[i] == 1:
            K_c2.append(kernel_matrix[:, i])

    # formatting matrices
    K_c1 = np.array(K_c1).T
    K_c2 = np.array(K_c2).T

    m1 = np.mean(K_c1, axis=1)
    m2 = np.mean(K_c2, axis=1)

    M = np.outer(m1-m2, m1-m2)

    n_1 = K_c1.shape[1]
    n_2 = K_c2.shape[1]

    I_n1 = np.identity(n_1) - (1 / n_1) * np.ones(n_1)
    I_n2 = np.identity(n_2) - (1 / n_2) * np.ones(n_2)

    N1 = K_c1.dot(I_n1.dot(K_c1.T))
    N2 = K_c2.dot(I_n2.dot(K_c2.T))

    N = N1 + N2

    u, v = np.linalg.eig(np.linalg.pinv(N).dot(M))

    return u.real, v.real, kernel_matrix


def l2_norm(a, b):
    return math.sqrt(np.sum((a - b) ** 2))


if __name__ == '__main__':
    filename = sys.argv[1]

    if len(sys.argv) > 2:
        spread_provided = True
    else:
        spread_provided = False

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data = pandas_dataframe.to_numpy()

    class_column = data[:1000, 0, np.newaxis]
    binary_data = np.where(class_column <= 50, class_column, 1)
    binary_data = np.where(binary_data == 1, binary_data, 0)

    data_matrix = data[:1000, 1:]

    if spread_provided:  # <ex> assign.py file.csv 100
        spread = float(sys.argv[2])
        u, v, kernel_matrix = find_best_spread(data_matrix, binary_data, spread)

        best_spread = spread
    else:  # <ex> assign.py file.csv
        _spread = [0.01, 0.1, 1, 10, 100]
        best_eigval_list = []  # choose the best spread based on the highest given eigenvalue

        # test with different Gaussian spread (sigma^2) values
        for spread_val in _spread:
            u, v, kernel_matrix = find_best_spread(data_matrix, binary_data, spread_val)
            # first eigenvalue -- corresponding with the most dominant eigenvector
            first_eigval = u[0]
            print("Spread val {} - first eigenvalue: {}\n".format(spread_val, first_eigval))
            best_eigval_list.append(first_eigval)

        best_spread = _spread[best_eigval_list.index(max(best_eigval_list))]

        u, v, kernel_matrix = find_best_spread(data_matrix, binary_data, best_spread)

    a = v[:, 0, np.newaxis].real

    # normalize
    a = a / math.sqrt(a.T.dot(kernel_matrix).dot(a))

    print("vector a:")
    print(a.T)

    # class-wise data projection on discriminant direction
    c1_projections = []
    c2_projections = []
    for i in range(0, binary_data.shape[0]):
        if binary_data[i] == 0:
            c1_projections.append(a.T.dot(kernel_matrix[:, i]))
        elif binary_data[i] == 1:
            c2_projections.append(a.T.dot(kernel_matrix[:, i]))
    c1_projections = np.array(c1_projections)
    c2_projections = np.array(c2_projections)

    plt.plot(c1_projections, np.zeros(c1_projections.shape[0]), 'o', color='red', markersize=3)
    plt.plot(c2_projections, np.zeros(c2_projections.shape[0]), 'x', color='blue', markersize=3)
    plt.title("data projection on the discriminant direction, spread = {}".format(best_spread))
    plt.xlabel('scalar projection values')
    plt.savefig("data_projections.png")
    plt.clf()

    sys.exit()
