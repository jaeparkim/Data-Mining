import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=3)
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


if __name__ == '__main__':
    
    '''filename = sys.argv[1]

    pandas_dataframe = pd.read_csv(filename)
    pandas_dataframe = pandas_dataframe.drop(['date', 'rv2'], axis=1)
    data_matrix = pandas_dataframe.to_numpy()
    '''
    """
    part 2
    """
    d_list = [100]
    iterations = 100

    for d in d_list:
        angle_values = []

        for i in range(int(iterations / 2)):
            rand_matrix = np.random.randint(2, size=(2, d))
            # rand_matrix = np.where(rand_matrix != 0, rand_matrix, -1)

            unit_vector_1 = rand_matrix[0, :] / np.linalg.norm(rand_matrix[0, :])
            unit_vector_2 = rand_matrix[1, :] / np.linalg.norm(rand_matrix[1, :])

            angle = np.degrees(np.arccos(np.dot(unit_vector_1, unit_vector_2)))

            angle_values.append(angle)

        num_bins = 180
        n, bins, patches = plt.hist(angle_values, num_bins, facecolor='blue', alpha=0.8, density=True)
        plt.xlabel('Angle between two random half-diagonals of dimension %d' % d)
        plt.ylabel('occurrence out of %d samples' % iterations)
        plt.title('number of bins: %d' % num_bins)
        plt.savefig("PMF_for_d=%d.png" % d)
        plt.clf()

        print("d = ", d)
        print("min: ", np.min(angle_values))
        print("max: ", np.max(angle_values))
        print("range: %f - %f" % (np.min(angle_values), np.max(angle_values)))
        print("mean: ", np.mean(angle_values))
        print("variance: ", np.var(angle_values))
        print("\n")

    sys.exit()
