import sys
import math
import random
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
plt.style.use('seaborn-whitegrid')


# save numpy array as a file
def save_np_arr(tempfile, content):
    with open(tempfile, 'wb') as f:
        np.save(f, content)

# open numpy array from a file
def open_np_arr(tempfile):
    with open(tempfile, 'rb') as f:
        a = np.load(f)
    return a

def linear_kernel(data_matrix_1, data_matrix_2):
    return np.dot(data_matrix_1, data_matrix_2.T)

def gaussian_kernel(data_matrix, spread):
    _norm = squareform(pdist(data_matrix, 'euclidean'))
    g_kernel = np.exp(-(_norm ** 2) / (2 * spread))
    return g_kernel

# gaussian kernel of data matrix 1 against 2
# e.g, test data against training data
def _gaussian_kernel(data_matrix_1, data_matrix_2, spread):
    kernel_matrix = np.zeros((len(data_matrix_1), len(data_matrix_2)))
    for i in range(len(data_matrix_1)):
        for j in range(len(data_matrix_2)):
            numer = np.linalg.norm(data_matrix_1[i] - data_matrix_2[j]) ** 2
            denom = float(2 * (spread))
            kernel_matrix[i][j] = math.exp(-numer/denom)
    return kernel_matrix

def prediction(svm_vector, y, kernel):
    x = np.sum((svm_vector * y * kernel), axis = 1)
    predictions_vector = np.sign(x)
    for prediction in predictions_vector:
        if prediction == 0:
            prediction = 1
    return predictions_vector

# vector_1: class-wise prediction
# vector_2: class-wise real data
def accuracy(alpha, vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        print("ERROR: prediction values have different shape from the actual")
        return -1
    else:
        num_good_predictions = 0
        not_svm_count = 0
        for i in range(len(vector_1)):
            # alpha[i] != 0 is the locations of support vector
            if vector_1[i] == vector_2[i] and alpha[i] != 0:
                num_good_predictions += 1
            elif vector_1[i] != vector_2[i] and alpha[i] != 0:
                continue
            else:  # alpha[i] == 0
                not_svm_count += 1

        _accuracy = num_good_predictions / (len(vector_1) - not_svm_count)
        return _accuracy

def get_weight_vec_and_bias(alpha, y, data_matrix, C):
    _alpha = np.copy(alpha)
    _data_matrix = np.copy(data_matrix)

    for val in _alpha:
        if 0 > val:
            val = 0

    # augment data matrix
    ones = np.ones((_data_matrix.shape[0], 1))
    x = np.append(ones, _data_matrix, axis=1)

    # weight vector = sum (Lagrange multiplier * class-wise target * data vector)
    w = np.sum((_alpha * y) * x.T, axis=1)

    b = w[0]

    num_support_vecs = 0
    for val in _alpha:
        if val < C and val != 0:
            num_support_vecs += 1

    return w, b , num_support_vecs

def SVM(data_matrix, y, kernel, kernel_param, C, eps, maxiter, loss):
    _maxiter = maxiter
    spread = kernel_param

    if kernel == 'linear': # hinge loss
        K = linear_kernel(data_matrix, data_matrix)

    elif kernel == 'gaussian': # hinge loss
        K = gaussian_kernel(data_matrix, spread=spread)
    
    elif kernel == 'polynomial': # quadratic loss
        # is not required to implement - undergrad
        return

    if loss == 'quadratic':
        delta = np.identity(K.shape[0])
        K = K - delta / 2*C

    K = K + 1

    eta = [1/_eta for _eta in np.diag(K)]
    eta = np.asarray(eta) # step size

    # iteration
    l1_norm = sys.maxsize
    alpha = np.zeros(K.shape[0])
    data_index = [i for i in range(K.shape[0])]

    while eps < l1_norm and _maxiter != 0:
        alpha_prev = np.copy(alpha)

        # grab a random item with k_index from data
        random.shuffle(data_index)

        for k_index in data_index:
            alpha[k_index] = alpha[k_index] + eta[k_index] * \
                (1 - y[k_index] * np.sum(np.multiply(np.multiply(alpha, y), K[:, k_index])))
            if alpha[k_index] < 0:
                alpha[k_index] = 0
            elif alpha[k_index] > C and loss == "hinge":
                alpha[k_index] = C

        _maxiter -= 1
        l1_norm = norm((alpha - alpha_prev), 1)

    print("SVM converged after %d iterations" % (maxiter-_maxiter))

    if kernel == 'linear':
        w, b, num_support_vecs = get_weight_vec_and_bias(alpha, y, data_matrix, C)
        print("weight vector:")
        print(w)
        print("bias:")
        print(b)
        print("Number of Support Vectors:", num_support_vecs)

    return alpha


'''
Dual Support Vector Machine (SVM) with Stochatic Gradient Ascent
'''
if __name__ == '__main__':
    filename = sys.argv[1]
    C = float(sys.argv[2])  # regulariation constant
    eps = float(sys.argv[3])  # convergence threshold
    maxiter = int(sys.argv[4])  # number of maximum permitted iterations
    kernel = sys.argv[5]  # "linear" or "gaussian"
    kernel_param = float(sys.argv[6])

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data = pandas_dataframe.to_numpy()

    class_column = data[:12000, 0]
    binary_data = np.where(class_column <= 50, class_column, -1)
    binary_data = np.where(binary_data == -1, binary_data, 1)

    data_matrix = data[:, 1:]
    scalar = StandardScaler()
    scalar.fit(data_matrix)
    data_matrix = scalar.transform(data_matrix)

    training = data_matrix[0:5000]
    validation = data_matrix[5000:7000]
    test = data_matrix[7000:12000]

    training_y = binary_data[0:5000]
    validation_y = binary_data[5000:7000]
    test_y = binary_data[7000:12000]

    alpha = SVM(training, training_y, kernel, kernel_param, C, eps, maxiter, loss='hinge')

    # save_np_arr('alpha_{0}.npy'.format(C), alpha)
    # alpha = open_np_arr('alpha_{0}.npy'.format(C))
    # alpha = open_np_arr('alpha_{}.npy'.format(C))


    # ignore below 0 or above C values
    for val in alpha:
        if 0 > val or val > C:
            val = 0

    # 2000 x 5000 = 2000 x 5000 * 5000 * 5000
    if kernel == 'linear':
        validation_K = linear_kernel(validation, training)
        test_K = linear_kernel(test, training)
    elif kernel == 'gaussian':
        validation_K = _gaussian_kernel(validation, training, spread=kernel_param)
        test_K = _gaussian_kernel(test, training, spread=kernel_param)
    

    valset_predictions = prediction(alpha, training_y, validation_K)
    print("Validation set accuracy: %.2f%%" % (accuracy(alpha, valset_predictions, validation_y) * 100))

    testset_predictions = prediction(alpha, training_y, test_K)
    print("Test set accuracy: %.2f%%" % (accuracy(alpha, testset_predictions, test_y) * 100))

    sys.exit()
