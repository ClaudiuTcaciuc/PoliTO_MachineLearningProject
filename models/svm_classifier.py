import numpy as np
import scipy.optimize as opt
import os

class SVMClassifier:
    def __init__(self, C=10.0, K=1.0):
        self.C = C
        self.K = K
        self.weight = None
        self.bias = None
    
    def fit(self, X, y, folder=None, test_only=False):
        
        folder = os.path.join("__results__", "svm", "Linear", folder) if folder else None
        
        if folder is not None and not os.path.exists(folder):
            os.makedirs(folder)
            # print(f"Folder {folder} created successfully.")
        
        weight_path = os.path.join(folder, "weight.npy") if folder else None
        bias_path = os.path.join(folder, "bias.npy") if folder else None
        
        if test_only and weight_path and bias_path and os.path.exists(weight_path) and os.path.exists(bias_path):
            self.weight = np.load(weight_path)
            self.bias = np.load(bias_path)
            print("Model loaded successfully.")
        else:
            if X.shape[0] > X.shape[1]:
                X = X.T
            
            d, n = X.shape # d = number of features, n = number of samples
            zi = 2 * y - 1
            
            X_tilde = np.vstack([X, np.ones((1, n)) * self.K])
            hessian = np.dot(X_tilde.T, X_tilde) * np.outer(zi, zi)
            
            def objective(alpha):
                Ha = hessian @ alpha
                loss = 0.5 * alpha @ Ha - np.sum(alpha)
                return loss
            def gradient(alpha):
                return hessian @ alpha - np.ones(n)
            
            bounds = [(0, self.C) for _ in range(n)]
            alpha_opt, _, _ = opt.fmin_l_bfgs_b(objective, np.zeros(n), fprime=gradient, bounds=bounds, factr=1.0)
            w_hat = np.sum(alpha_opt * zi * X_tilde, axis=1)
            
            self.weight = w_hat[:-1]
            self.bias = w_hat[-1]* self.K
            dual_loss = -objective(alpha_opt)
            # print(f"Optimal dual loss: {dual_loss}")
            
            def primal_loss(w_hat):
                reg = 0.5 * np.linalg.norm(w_hat) ** 2
                hinge_loss = np.maximum(0, 1 - zi * (w_hat.T @ X_tilde))
                return reg + self.C * np.sum(hinge_loss)
            
            primal_loss_value = primal_loss(w_hat)
            # print(f"Primal loss: {primal_loss_value}")
            duality_gap = primal_loss_value - dual_loss
            # print(f"Duality gap: {np.abs(duality_gap)}")
            
            if weight_path:
                np.save(weight_path, self.weight)
                # print("Weights saved successfully.")
            if bias_path:
                np.save(bias_path, self.bias)
                # print("Bias saved successfully.")

    def score(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        if self.weight is None:
            raise ValueError("Model not trained yet. Call fit() before score().")
        
        return self.weight.T @ X + self.bias
    
    def predict(self, X, y):
        z = self.score(X)
        y_pred = (z > 0) * 1
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy

    def get_weight(self):
        return self.weight
    
    def get_bias(self):
        return self.bias

    def set_weight(self, weight):
        self.weight = weight
        
    def set_bias(self, bias):
        self.bias = bias
    
    def set_C(self, C):
        self.C = C
        
    def get_C(self):
        return self.C
    
    def set_K(self, K):
        self.K = K
        
    def get_K(self):
        return self.K
    