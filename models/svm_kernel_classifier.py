import numpy as np
import scipy.optimize as opt
import os

class SVMClassifierKernel:
    def __init__(self, C=1.0, eps=1.0, kernel=None):
        self.kernel = kernel
        self.eps = eps
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
    
    def get_kernel_name(self):
        return "Custom"
    
    def fit(self, X, y, folder=None, test_only=False):
        
        folder = os.path.join("__results__", "svm", self.get_kernel_name(), folder) if folder else None
        
        if folder is not None and not os.path.exists(folder):
            os.makedirs(folder)
            # print(f"Folder {folder} created successfully.")
        
        alpha_path = os.path.join(folder, "alpha.npy") if folder else None
        support_vectors_path = os.path.join(folder, "support_vectors.npy") if folder else None
        support_vector_labels_path = os.path.join(folder, "support_vector_labels.npy") if folder else None
        
        if test_only and alpha_path and support_vectors_path and support_vector_labels_path and os.path.exists(alpha_path) and os.path.exists(support_vectors_path) and os.path.exists(support_vector_labels_path):
            self.alpha = np.load(alpha_path)
            self.support_vectors = np.load(support_vectors_path)
            self.support_vector_labels = np.load(support_vector_labels_path)
            print("Model loaded successfully.")
        else:
            if X.shape[0] > X.shape[1]:
                X = X.T
            
            d, n = X.shape # d = number of features, n = number of samples
            
            zi = 2 * y - 1
            K = self.kernel(X, X) + self.eps
            H = np.outer(zi, zi) * K
            
            def objective(alpha):
                Ha = H @ alpha
                loss = 0.5 * alpha @ Ha - np.sum(alpha)
                return loss
            def gradient(alpha):
                return H @ alpha - np.ones(n)
            
            bounds = [(0, self.C) for _ in range(n)]
            alpha, _, _ = opt.fmin_l_bfgs_b(objective, np.zeros(n), fprime=gradient, bounds=bounds, factr=1.0)
            support_mask = alpha > 0
            
            self.alpha = alpha[support_mask]
            self.support_vectors = X[:, support_mask]
            self.support_vector_labels = zi[support_mask]
            # print(f"SVM Kernel dual loss: {-objective(alpha):.4f}")
            
            if alpha_path:
                np.save(alpha_path, self.alpha)
            if support_vectors_path:
                np.save(support_vectors_path, self.support_vectors)
            if support_vector_labels_path:
                np.save(support_vector_labels_path, self.support_vector_labels)
    
    def score(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        if self.alpha is None:
            raise ValueError("Model not trained yet. Call fit() before score().")
        
        n = X.shape[1]
        K = self.kernel(X, self.support_vectors) + self.eps
        z = np.sum(self.alpha * self.support_vector_labels * K, axis=1)
        return z

    def predict(self, X, y):
        z = self.score(X)
        y_pred = (z > 0) * 1
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy

    def get_eps(self):
        return self.eps
    
    def set_eps(self, eps):
        self.eps = eps
        
    def get_C(self):
        return self.C
    
    def set_C(self, C):
        self.C = C
        
    def get_alpha(self):
        return self.alpha
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def get_support_vectors(self):
        return self.support_vectors
    
    def set_support_vectors(self, support_vectors):
        self.support_vectors = support_vectors
    
    def get_support_vector_labels(self):
        return self.support_vector_labels
    
    def set_support_vector_labels(self, support_vector_labels):
        self.support_vector_labels = support_vector_labels

class SVMClassifierPolyKernel(SVMClassifierKernel):
    def __init__(self, C=1.0, eps=1.0, degree=2, delta=1.0):
        kernel_func = self.kernel_func
        super().__init__(C, eps, kernel_func)
        self.degree = degree
        self.delta = delta
    
    def kernel_func(self, X1, X2):
        return (np.dot(X1.T, X2) + self.delta) ** self.degree
    
    def get_degree(self):
        return self.degree
    
    def set_degree(self, degree):
        self.degree = degree
        
    def get_bias(self):
        return self.delta
    
    def set_bias(self, delta):
        self.delta = delta

    def get_kernel_name(self):
        return "Polynomial"

class SVMClassifierRBFKernel(SVMClassifierKernel):
    def __init__(self, C=1.0, eps=1.0, gamma=1.0):
        kernel_func = self.kernel_func
        super().__init__(C, eps, kernel_func)
        self.gamma = gamma
        
    def kernel_func(self, X1, X2):
        return np.exp(-self.gamma * np.sum((X1[:, :, np.newaxis] - X2[:, np.newaxis, :])**2, axis=0))
    
    def get_gamma(self):
        return self.gamma
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def get_kernel_name(self):
        return "RBF"