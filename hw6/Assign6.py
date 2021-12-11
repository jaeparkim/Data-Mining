import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
plt.style.use('seaborn-whitegrid')


def cov_matrix(data_matrix, mean_vector):
    centered_data = data_matrix - mean_vector
    covariance_mat = np.dot(np.transpose(centered_data), centered_data) / centered_data.shape[0]
    return covariance_mat


def diag_cov_mat(cov_matrix):
    diagonal_values = []
    for i in range(0, cov_matrix.shape[0]):
        diagonal_values.append(cov_matrix[i][i])
    
    A = np.asarray(diagonal_values)
    _cov_matrix = linalg.block_diag(*A)

    return _cov_matrix


def testset_evaluation(test_response, predictions_list):
    num_correct_predictions = 0
    c_0, c_1, c_2, c_3 = 0, 0, 0, 0

    for idx, val in enumerate(predictions_list):
        if val == test_response[idx]:
            num_correct_predictions += 1
            if val == 0:
                c_0 += 1
            elif val == 1:
                c_1 += 1
            elif val == 2:
                c_2 += 1
            else:
                c_3 += 1

    print("total accuracy: %f" %(num_correct_predictions / test_response.shape[0]))
    print("class specific accuracy")
    print("\tClass 0: %f" % (c_0 / (predictions_list == 0).sum()))
    print("\tClass 1: %f" % (c_1 / (predictions_list == 1).sum()))
    print("\tClass 2: %f" % (c_2 / (predictions_list == 2).sum()))
    print("\tClass 3: %f" % (c_3 / (predictions_list == 3).sum()))
    print("class-wise recall values")
    print("\tClass 0: %f" % (c_0 / (test_response == 0).sum()))
    print("\tClass 1: %f" % (c_1 / (test_response == 1).sum()))
    print("\tClass 2: %f" % (c_2 / (test_response == 2).sum()))
    print("\tClass 3: %f" % (c_3 / (test_response == 3).sum()))

def bayes_classifier_prediction(classwise_data_list, prior_probs, option):
    data_c_0 = classwise_data_list[0]
    data_c_1 = classwise_data_list[1]
    data_c_2 = classwise_data_list[2]
    data_c_3 = classwise_data_list[3]

    # class-wise mean points
    mean_points = []
    mean_points.append(data_c_0.sum(axis=0) / data_c_0.shape[0])
    mean_points.append(data_c_1.sum(axis=0) / data_c_1.shape[0])
    mean_points.append(data_c_2.sum(axis=0) / data_c_2.shape[0])
    mean_points.append(data_c_3.sum(axis=0) / data_c_3.shape[0])

    # covariance matrices
    cov_matrix_0 = cov_matrix(data_c_0, mean_points[0])
    cov_matrix_1 = cov_matrix(data_c_1, mean_points[1])
    cov_matrix_2 = cov_matrix(data_c_2, mean_points[2])
    cov_matrix_3 = cov_matrix(data_c_3, mean_points[3])

    if option == 'naive_bayes':
        cov_matrix_0 = diag_cov_mat(cov_matrix_0)
        cov_matrix_1 = diag_cov_mat(cov_matrix_1)
        cov_matrix_2 = diag_cov_mat(cov_matrix_2)
        cov_matrix_3 = diag_cov_mat(cov_matrix_3)

    # make predictions on test data and compare with test set response
    predictions_list = []
    for _, data in enumerate(test_data):
            # probability of class k given a data point
            p_0 = stats.multivariate_normal.pdf(
                data, mean_points[0], cov_matrix_0) / prior_probs[0]
            p_1 = stats.multivariate_normal.pdf(
                data, mean_points[1], cov_matrix_1) / prior_probs[1]
            p_2 = stats.multivariate_normal.pdf(
                data, mean_points[2], cov_matrix_2) / prior_probs[2]
            p_3 = stats.multivariate_normal.pdf(
                data, mean_points[3], cov_matrix_3) / prior_probs[3]
            _max = max(p_0, p_1, p_2, p_3)
            if _max == p_0:
                predictions_list.append(0)
            elif _max == p_1:
                predictions_list.append(1)
            elif _max == p_2:
                predictions_list.append(2)
            else:
                predictions_list.append(3)

    predictions_list = np.asarray(predictions_list)

    return predictions_list

'''
full Bayes classifier and naive Bayes classifier
'''
if __name__ == '__main__':
    filename = sys.argv[1]

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data = pandas_dataframe.to_numpy()

    response = data[:, 0]
    data_matrix = data[:, 1:]

    # class value conversion for multiclass classification
    for idx, val in enumerate(response):
        if val <= 40:
            response[idx] = 0
        elif 40 < val <= 60:
            response[idx] = 1
        elif 60 < val <= 100:
            response[idx] = 2
        else:
            response[idx] = 3

    training_data = data_matrix[0:14735] # 14735 data points
    test_data = data_matrix[14735:19735] # 5000 data points

    training_response = response[0:14735, np.newaxis]
    test_response = response[14735:19735, np.newaxis]

    # prior probability
    prior_probs = []
    for i in range(0, 4):
        classwise_prob = (training_response == i).sum() / training_response.shape[0]
        prior_probs.append(classwise_prob)

    # organize and group data by each class
    data_c_0, data_c_1, data_c_2,data_c_3 = [], [], [], []
    for idx, data in enumerate(training_data):
        classwise_classification = training_response[idx]
        if classwise_classification == 0:
            data_c_0.append(data)
        elif classwise_classification == 1:
            data_c_1.append(data)
        elif classwise_classification == 2:
            data_c_2.append(data)
        else:  # classwise_classification == 3
            data_c_3.append(data)

    data_c_0 = np.asarray(data_c_0)
    data_c_1 = np.asarray(data_c_1)
    data_c_2 = np.asarray(data_c_2)
    data_c_3 = np.asarray(data_c_3)
    classwise_data_list = [data_c_0, data_c_1, data_c_2, data_c_3]


    '''
    running the Bayes classification algorithm
    '''

    predictions_list = bayes_classifier_prediction(classwise_data_list, prior_probs, "bayes")
    print("\n>>>>>>>>>>>> Bayes Classifier <<<<<<<<<<<<")
    testset_evaluation(test_response, predictions_list)


    predictions_list = bayes_classifier_prediction(classwise_data_list, prior_probs, "naive_bayes")
    print("\n>>>>>>>>> naive Bayes Classifier <<<<<<<<<")
    testset_evaluation(test_response, predictions_list)


    sys.exit()
