import sys
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.spatial import distance

from tqdm import tqdm


def EM(data, mu, k, eps, ridge, maxiter):
    sigma_vals = [np.identity(len(data[0])) for i in range(k)]
    posterior_probs = [1/k for i in range(k)]
    w = np.random.rand(k, len(data))
    new_mu = np.copy(mu)
    
    for l in range(maxiter):
        w = update_normalized_weights(data, mu, sigma_vals, posterior_probs, w, k, ridge)
        new_mu, sigma_vals, posterior_probs = reestimate_params(data, k, w)
        diff = 0
        for i in range(k):
            diff += np.linalg.norm(new_mu[i] - mu[i], 2)
        
        if diff < eps:
            return new_mu, sigma_vals, posterior_probs, w

        mu = new_mu

                
def update_normalized_weights(data, mus, sigmas, posterior_probs, w, k, ridge):
    for j in range(len(data)):
        f_val = [multivariate_normal.logpdf(
            data[j], mus[i], sigmas[i], allow_singular=True) for i in range(k)]

        denomenator = 0
        for i in range(k):
            denomenator += f_val[i] * posterior_probs[i]
        for i in range(k):
            numerator = (f_val[i] * posterior_probs[i])
            w[i][j] = numerator / denomenator

    return w


def reestimate_params(data, k, w):
    new_means = []
    new_covs = []
    new_priors = []

    for i in range(k):

        # reestimate means
        numer = 0
        for j in range(len(data)):
            numer += w[i][j] * data[j]
        new_means.append(numer / np.sum(w[i]))

        # reestimate sigma values
        sum = 0
        for j in range(len(data)):
            vec = np.reshape(np.asarray(data[j] - new_means[i]), (26, 1))
            norm_vec = np.dot(vec, vec.T)
            sum += w[i][j]*(norm_vec)
        new_covs.append(sum / np.sum(w[i]))

        # reestimate prior probabilities
        new_priors.append(np.sum(w[i]) / len(data))

    return new_means, new_covs, new_priors


def find_farthest_means(data, k):
    mean_vals = []
    new_mean_vals = []

    init_mean_idx = np.random.randint(len(data))
    mean_vals.append(np.reshape(np.asarray(data[init_mean_idx]), (1, 26)))
    new_mean_vals.append(data[init_mean_idx].tolist())

    distances = []
    for i in range(k - 1):
        distances.append(distance.cdist(
            mean_vals[i], data, 'euclidean')[0].tolist())

        if len(distances) != 1:
            min_dists = []
            for dist in np.transpose(distances):
                min_dists.append(min(dist))
            next_index = min_dists.index(max(min_dists))
        else:
            next_index = distances[0].index(max(distances[0]))

        mean_vals.append(np.reshape(np.asarray(data[next_index]), (1, 26)))
        new_mean_vals.append(data[next_index].tolist())

    return new_mean_vals


def purity_vals(data, response, w):
    prev = [0, 0, 0, 0]
    classwise_purity = [0, 0, 0, 0]
    weights = np.transpose(w)

    for i, weight in enumerate(weights):
        weight = weight.tolist()
        cl = weight.index(max(weight))
        prev[cl] += 1
        if response[i] == cl:
            classwise_purity[cl] +=1

    val = np.sum(classwise_purity) / len(data)

    return prev, val

def pretty_print(k, mean, sigma_vals, cluster_sizes, purity_val):
    print("Classwise mean vectors:")
    for i in range(k):
        print(str(i) + ": " + str(mean[i])) 
    print("\nClasswise covariance matrix:")
    for i in range(k):
        print(str(i) + ": " + str(sigma_vals[i])) 
    print("\nClasswise cluster size:")
    for i in range(k):
        print("class" + str(i) + ": " + str(cluster_sizes[i])) 
    print("\nPurity Score: " + str(purity_val))
     

'''
Expectation Maximization Clustering
'''
if __name__ == '__main__':
    filename = sys.argv[1]
    k = int(float(sys.argv[2]))
    eps = float(sys.argv[3])
    ridge = float(sys.argv[4])
    maxiter = int(float(sys.argv[5]))

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


    mu = find_farthest_means(data_matrix, k)

    mu, sigmas, p_hats, w = EM(data_matrix, mu, k, eps, ridge, maxiter)

    cluster_sizes, p_score = purity_vals(data_matrix, response, w)

    pretty_print(k, mu, sigmas, cluster_sizes, p_score)

    sys.exit()
    