import numpy as np
import scipy

def pca(data, n_features=4, required_eigen_vectors=False):
    """ Compute the PCA decomposition on n features
        Data: (n_features, n_samples)
    """
    shape_changed = False
    
    if data.shape[0] > data.shape[1]:
        data = data.T
        shape_changed = True

    mean = np.mean(data, axis=1).reshape(-1, 1)
    centered_data = data - mean
    cov = np.dot(centered_data, centered_data.T) / data.shape[1]
    
    eigen_values, eigen_vectors = scipy.linalg.eigh(cov)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vectors[:, sorted_index]
    selected_eigen_vectors = sorted_eigen_vectors[:, :n_features]
    
    new_data = np.dot(selected_eigen_vectors.T, data)
    
    # return the data in the original shape
    if shape_changed:
        new_data = new_data.T
    
    if required_eigen_vectors:
        return new_data, selected_eigen_vectors
    return new_data

def compute_Sw_Sb(data, label):
    """ Compute the within-class and between
            Sw: within-class scatter matrix
                formula: sum(Cov(Xi) for i in classes)
            Sb: between-class scatter matrix
                formula: sum(Ni * (mean(Xi) - mean(X)) for i in classes)
    """
    data_class = [data[:, label==i] for i in np.unique(label)]
    sample_class = [data_class[i].shape[1] for i in np.unique(label)]
    
    mean = np.mean(data, axis=1).reshape(-1, 1)
    mean_class = [np.mean(data_class[i], axis=1).reshape(-1, 1) for i in np.unique(label)]
    S_w, S_b = 0, 0
    for i in np.unique(label):
        data_c = data_class[i] - mean_class[i]
        cov_c = np.dot(data_c, data_c.T) / data_c.shape[1]
        S_w += sample_class[i] * cov_c
        diff = mean_class[i] - mean
        S_b += sample_class[i] * np.dot(diff, diff.T)
    S_w /= data.shape[1]
    S_b /= data.shape[1]
    return S_w, S_b

def lda(data, label, n_features=3, required_eigen_vectors=False):
    """ Compute the LDA decomposition on n features
            n_max = n_features - 1
        Data: (n_features, n_samples)
    """
    shape_changed = False
    
    if data.shape[0] > data.shape[1]:
        data = data.T
        shape_changed = True
    
    if n_features > len(np.unique(label)) - 1:
        raise ValueError(f"n_features must be less than {len(np.unique(label)) - 1}")
    
    Sw, Sb = compute_Sw_Sb(data=data, label=label)
    
    _, eigen_vectors = scipy.linalg.eigh(Sb, Sw)
    selected_eigen_vectors = eigen_vectors[:, ::-1][:, :n_features]
    lda_data = np.dot(selected_eigen_vectors.T, data)
    
    # return the data in the original shape
    if shape_changed:
        lda_data = lda_data.T
    
    if required_eigen_vectors:
        return lda_data, selected_eigen_vectors
    return lda_data

def standardize(data: np.array, return_params=False):
    """ Standardize the data """
    mean = np.mean(data, axis=1).reshape(-1, 1)
    std = np.std(data, axis=1).reshape(-1, 1)
    if return_params:
        return (data - mean) / std, mean, std
    return (data - mean) / std

def normalize(data: np.array):
    """ Normalize the data """
    min_val = np.min(data, axis=1).reshape(-1, 1)
    max_val = np.max(data, axis=1).reshape(-1, 1)
    return (data - min_val) / (max_val - min_val)