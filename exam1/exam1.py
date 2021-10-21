import numpy as np
import matplotlib.pyplot as plt
import math

three = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
    ]
two = [
    [1, 1],
    [1, 1],
    ]
four = [
    [1, 3, 2],
    [2, 2, 4],
    [2, 1, 5],
    [3, 4, 5]
    ]

def mean_vector(data_matrix, num_rows):
    """
    A Method used to calculate the mean vector mu along each column
    ...
    Parameters
    ---------
    data_matrix : np.ndarray
        A matrix containing the data for which the mean vector is to be calculated
    
    num_rows : int
        The number of rows in your data matrix
    """
    sum_vector = np.zeros(27)
    for row in data_matrix:
        for i, item in enumerate(row):
            sum_vector[i] = sum_vector[i] + item
            
    mean_vector = sum_vector/num_rows
    return mean_vector

def center_data(data_matrix, mean_vector):
    """
    Given the data matrix and the mean vector, returns the data values normalized to the mean
    Parameters
    --------
    data_matrix: np.ndarray
        A matrix containing your data
    mean_vector: np.ndarray
        The vector containing the average value for each column in the data matrix    
    """
    centered_data = data_matrix - mean_vector

    return centered_data

def total_variance(data_matrix, num_rows):
    """
    Calcualtes the total variance of a data set
    ...
    Parameters
    --------
    data_matrix: np.ndarray
        A matrix containing your data
    num_rows: int
        The number of rows in your data matrix
    """
    total_variance = 0
    for row in data_matrix:
        total_variance = total_variance + (np.linalg.norm(row) ** 2)

    total_variance = total_variance/(num_rows-1)

    return total_variance


def cov_matrix_inner(data_matrix, num_rows):
    """
    Uses the inner product to calculate the covariance matrix for your data set
    ...
    Parameters
    ----------
    data_matrix: np.ndarray
        A matrix containing your data
    num_rows: int
        The number of rows in your data matrix
    """
    cov_matrix_inner = np.dot(np.transpose(data_matrix), data_matrix)
    cov_matrix_inner = cov_matrix_inner / num_rows

    return cov_matrix_inner

def cov_matrix_outer(data_matrix, num_rows):
    """
    Uses the outer product to calculate the covariance matrix of your data set
    ...
    Parameters
    ---------
    data_matrix: np.ndarray
        A matrix containing your data
    num_rows: int
        The number of rows in your data matrix
    """
    cov_matrix_outer = np.zeros((27,27))
   
    for i,row in enumerate(data_matrix):
        #print((np.dot(np.transpose(row),row)) )
        #print(row)
        cov_matrix_outer = cov_matrix_outer + (np.outer(row, np.transpose(row))) 
    cov_matrix_outer = cov_matrix_outer / num_rows
    return cov_matrix_outer

def correlation_matrix(data_matrix, num_rows):
    """
    Calculates the Correlation Matrix across each attribute in your data matrix
    ...
    Parameters
    --------
    data_matrix: np.ndarray
        A matrix containing your data
    """
    transposed_data = np.transpose(data_matrix)
    correlation_matrix = np.zeros((27,27))

    for i in range(27):
        for j in range(i,27):
            correlation_matrix[i][j] = np.dot(transposed_data[i]/np.linalg.norm(transposed_data[i]), transposed_data[j]/np.linalg.norm(transposed_data[j]))
            correlation_matrix[j][i] = correlation_matrix[i][j]

    return correlation_matrix

def finding_interesting_points(correlation_matrix, data_matrix):
    """
    This method returns the indices of 3 interesting points in the following order: The least correlated attributes, the most positively correlated points, and the most negatively correlated points
    ...
    Parameters
    --------
    correlation_matrix: np.ndarray
        Matrix containing the correlation coefficients between each of the attributes in your data set
    """
    corr_abs = np.abs(correlation_matrix)
    least_corr = np.unravel_index(np.argmin(corr_abs), corr_abs.shape)

    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if i == j:
                correlation_matrix[i][j] = 0

    #print(correlation_matrix)

    #print(np.where(correlation_matrix < .99999, correlation_matrix, np.nan))
    #max_val_ind = np.unravel_index(np.argmax(np.where(correlation_matrix < 1, correlation_matrix, np.nan)), correlation_matrix.shape)
    max_val_ind = np.unravel_index(np.argmax(correlation_matrix), correlation_matrix.shape)
    min_val_ind = np.unravel_index(np.argmin(correlation_matrix), correlation_matrix.shape)


    

    #print(data_matrix[:,max_val_ind[0]].shape)
    
    plt.title("Most Positively Correlated Point")
    plt.scatter(data_matrix[:,max_val_ind[0]], data_matrix[:,max_val_ind[1]])
    plt.xlabel("Attribute " + str(max_val_ind[0]+1))
    plt.ylabel("Attribute " + str(max_val_ind[1]+1))
    plt.show()
    plt.title("Most Negatively Correlated Point")
    plt.scatter(data_matrix[:,min_val_ind[0]], data_matrix[:,min_val_ind[1]])
    plt.xlabel("Attribute " + str(min_val_ind[0]+1))
    plt.ylabel("Attribute " + str(min_val_ind[1]+1))
    plt.show()
    plt.title("Least Correlated Point")
    plt.scatter(data_matrix[:,least_corr[0]], data_matrix[:,least_corr[1]])
    plt.xlabel("Attribute " + str(least_corr[0]+1))
    plt.ylabel("Attribute " + str(least_corr[1]+1))
    plt.show()


    #print(correlation_matrix[max_val_ind[0]][max_val_ind[1]])
    


def power_iterative_method(covariance_matrix, data_matrix, eps):
    """
    This function uses the Power Iterative Method to find the Eigenvector of a data set using its covariance matrix
    ...
    Parameters
    --------
    covariance_matrix: np.ndarray
        Covariance matrix of your data set
    data_matrix: np.ndarray
        Matrix containing the your data
    eps: float
        Threshold between iterations of the Power Iterative Method to determine when the vector has converged to the eigenvector
    """
    x0 = np.random.rand(27,1)
    x1 = np.zeros(27)
    count = 0
    x_max = 0
    while True:
        x1 = np.matmul(covariance_matrix, x0)
        x_max = np.amax(np.abs(x1))
        x1 = x1/x_max
        diff = np.linalg.norm(x1-x0)
        x0 = x1
        if(diff < eps): break
    print(x1)
    print("Eigenvalue: " + str(print("{:.3f}".format(x_max))))

    projections = []
    for row in data_matrix:
        top = np.dot(row.T, x0)
        bottom = np.dot(x0.T, x0)
        projections.append(np.divide(top, bottom))

    fives = [5]*len(data_matrix)
    plt.title("Projection of Data Set onto The Eigenvector")
    plt.scatter(projections, fives)
    plt.show()
    
def projection(matrix, vector):
    matrix = center_data(matrix)
    vector = vector / np.linalg.norm(vector)
    projections = []
    for row in matrix:
        projections.append(np.dot(row, vector))
    
    return projections

def m_distance(matrix, covariance_matrix):
    matrix = center_data(matrix)
    point_one = matrix[]
    point_two = matrix[]

    distance = np.dot(point_one.T, np.linalg.inv(covariance_matrix))
    distance = np.dot(distance, point_two)

def cosine(vector1, vector2):
    cos = np.dot((vector1/np.linalg.norm(vector1)),(vector2/np.linalg.norm(vector2)))
    theta = math.acos(cos) * 180 / (math.pi)
    return theta


    
        




