import numpy as np
import scipy

def load_data():
    #TODO: make the path not hard coded
    path = './data/trainData.txt'
    
    data_matrix = np.loadtxt(path, delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt(path, delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

def load_validation_data():
    #TODO: make the path not hard coded
    path = './data/evalData.txt'
    
    data_matrix = np.loadtxt(path, delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt(path, delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

def split_data(data, label, perc=(2.0/3.0), seed=0):
    """ Split the data 2/3 for train and 1/3 for test """
    
    n_train = int(data.shape[1] * perc)
    np.random.seed(seed)
    index = np.random.permutation(data.shape[1])
    index_train = index[:n_train]
    index_test = index[n_train:]

    data_train = data[:, index_train]
    label_train = label[index_train]
    data_test = data[:, index_test]
    label_test = label[index_test]
    
    return data_train, label_train, data_test, label_test

def compute_statistics(data):
    """ Compute the mean, variance, std and covariance matrix of the data """
    mu_class = np.mean(data, axis=1).reshape(-1, 1)
    print(f'Empirical dataset mean\n{mu_class}')
    
    # Centered data
    centered_data = data - mu_class
    print(f'Centered data shape\n{centered_data.shape}')
    
    # Covariance matrix
    cov_matrix = np.cov(centered_data)
    print(f'Covariance matrix shape\n{cov_matrix.shape}')

    # Variance
    var = np.var(data, axis=1).reshape(-1, 1)
    print(f'Variance\n {var}')
    
    # std
    std = np.std(data, axis=1).reshape(-1, 1)
    print(f'Standard deviation\n{std}')
    
