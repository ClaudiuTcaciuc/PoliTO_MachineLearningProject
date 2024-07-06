import numpy as np
import scipy.optimize as opt
import os

class LogisticRegression:
    """ Logistic Regression Classifier
        The classifier is based on the sigmoid function
        The classifier assumes that the data is linearly separable
        The classifier assumes that the data is binary
        
        How to use:
        0. Change the label to -1 and 1
        1. Create an instance of the class and set the lambda parameter
        2. Call the fit method to train the model: fit(X, y)
        3. Call the predict method to compute the accuracy: predict(X, y)
    """
    def __init__(self, lambda_=0):
        self.weights = None
        self.lambda_ = lambda_
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def logreg_obj(self, v, X, y):
        
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        _, n = X.shape
        w, b = v[0:-1], v[-1]
        z = w.T @ X + b
        zi = 2 * y - 1 
        loss = np.logaddexp(0, -zi * z)
        reg = (self.lambda_ / 2) * np.sum(w ** 2)
        J = (1 / n) * np.sum(loss) + reg
        return J
    
    def fit(self, X, y, folder=None, test_only=False):
        
        folder = os.path.join("__results__", "logistic_regression", folder) if folder else None
        
        if folder is not None and not os.path.exists(folder):
            os.makedirs(folder)
            # print(f"Folder {folder} created successfully.")
            
        weights_path = os.path.join(folder, "weights.npy") if folder else None
        
        # if the folder is not empty and we are testing only then load the weights
        if test_only and weights_path and os.path.exists(weights_path):
            self.weights = np.load(weights_path)
            # print("Weights loaded successfully.")
        else:
            # Train the model
            if X.shape[0] > X.shape[1]:
                X = X.T
        
            d, _ = X.shape
            x0_train = np.zeros(d + 1)
            
            x_opt, value, _ = opt.fmin_l_bfgs_b(self.logreg_obj, x0_train, args=(X, y), approx_grad=True)
            # print(f"Optimal value: {value}")
            self.weights = x_opt
            
            if weights_path:
                np.save(weights_path, self.weights)
                # print("Weights saved successfully.")
    
    def score(self, X):
        
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit() before score().")
        
        z = self.weights[:-1].T @ X + self.weights[-1]
        return z
    
    def predict(self, X, y):
        z = self.score(X)
        probabilities = self.sigmoid(z)
        y_pred = np.where(probabilities > 0.5, 1, 0)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
    
    def get_lambda(self):
        return self.lambda_
    
    def set_weights(self, weights):
        self.weights = weights
    
    def get_weights(self):
        return self.weights

class LogisticRegressionWeighted(LogisticRegression):
    def __init__(self, lambda_, pi, n_T, n_F):
        super().__init__(lambda_)
        self.n_T = n_T
        self.n_F = n_F
        self.pi = pi
    
    def logreg_obj(self, v, X, y):

        if X.shape[0] > X.shape[1]:
            X = X.T
        
        _, n = X.shape
        w, b = v[0:-1], v[-1]
        z = w.T @ X + b
        zi = 2 * y - 1 
        loss = np.logaddexp(0, -zi * z)
        weights = np.where(zi > 0, self.pi / self.n_T, (1 - self.pi) / self.n_F)
        weighted_loss = weights * loss
        
        reg = (self.lambda_ / 2) * np.linalg.norm(w)**2
        J = np.sum(weighted_loss) + reg
        return J

class QuadraticExpansion(LogisticRegression):
    """ Quadratic Expansion for the data """
    @staticmethod
    def expand(X):
        # X: n_features, n_samples
        
        shape_changed = False
        if X.shape[0] > X.shape[1]:
            shape_changed = True
            X = X.T
        
        data_row = X.shape[0]
        data_col = X.shape[1]
        quad_features = np.zeros((data_row**2 + data_row, data_col))

        for i in range(data_col):
            tmp = np.dot(X[:, i].reshape(data_row, 1), X[:, i].reshape(1, data_row))
            quad_features[:data_row**2, i] = tmp.flatten()
            quad_features[data_row**2:, i] = X[:, i]

        if shape_changed:
            quad_features = quad_features.T
        
        return quad_features