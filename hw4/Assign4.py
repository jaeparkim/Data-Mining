import sys
import math
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


if __name__ == '__main__':
    filename = sys.argv[1]
    alpha = float(sys.argv[2])
    eta = float(sys.argv[3]) # step size
    eps = float(sys.argv[4]) # convergence threshold
    maxiter = int(sys.argv[5])

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data = pandas_dataframe.to_numpy()

    scalar = StandardScaler()
    scalar.fit(data)
    data = scalar.transform(data)

    y = data[:, 0, np.newaxis]
    data_matrix = data[:, 1:]

    augmented = np.ones([len(data_matrix), 1], dtype=float)
    data_matrix = np.append(augmented, data_matrix, axis=1)

    training = data_matrix[0:13735, :]
    validation = data_matrix[13735:15735, :]
    testing = data_matrix[15735:, :]

    training_y = y[0:13735, :]
    validation_y = y[13735:15735, :]
    testing_y = y[15735:, :]

    w = np.ones([1, training.shape[1]])
    l1_norm = sys.maxsize


    while eps < l1_norm and maxiter != 0:
        gradient_w = - np.dot(training.T, training_y) + np.dot(training.T, training).dot(w.T) + (alpha * w.T)

        w_new = w - eta * gradient_w.T

        l1_norm = np.linalg.norm((w- w_new))
        w = w_new
        maxiter -= 1
        if maxiter == 0:
            print("hit max iteration!")
            break
        
    print("w:")  
    print(w)
    print("\n\u03B7 (step size):", eta)
    print("\u03B5 (convergence threshold):", eps)


    closed_form_w = np.linalg.pinv(training.T.dot(training) + alpha * np.identity(training.shape[1])).dot(training.T).dot(training_y)
    # print(closed_form_w.T)
    

    # Square Sum Error on validation set
    Dw = validation.dot(w.T)
    SSE = np.square(validation_y - Dw)
    SSE = np.sum(SSE)
    print("SSE on validation set:", "{: .3f}".format(SSE))

    # Square Sum Error on test set
    Dw = testing.dot(w.T)
    SSE = np.square(testing - Dw)
    SSE = np.sum(SSE)
    print("SSE on test set:", "{: .3f}".format(SSE))


    sys.exit()
