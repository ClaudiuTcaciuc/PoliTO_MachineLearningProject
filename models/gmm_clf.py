import numpy as np
import scipy.special
import scipy.linalg
import os
import pickle

class GMM:
    def __init__(self, n_components=1, covariance_type='full', tol=1e-6, psiEig = None, lbgAlpha = 0.1, verbose=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.psiEig = psiEig
        self.lbgAlpha = lbgAlpha
        self.verbose = verbose
        self.gmm_list = []

    def log_gau_pdf(self, X, mu, sigma):
        """ Compute the log of the Gaussian pdf
            X: n_features, n_samples
            mu: class mean
            sigma: class covariance matrix
        """
        
        if X.shape[0] < X.shape[1]:
            X = X.T
        
        _, d = X.shape # n: samples, d: features,
        X_c = X - mu.reshape(1, -1)
        inv_sigma = np.linalg.inv(sigma)
        sign, log_det_sigma = np.linalg.slogdet(sigma)
        det_sign = sign * log_det_sigma
        quad_form = np.sum(X_c @ inv_sigma * X_c, axis=1)
        log_pdf = -0.5 * (d * np.log(2 * np.pi) + det_sign + quad_form)
        
        return log_pdf
    
    def vcol(self, x):
        return x.reshape((x.size, 1))

    def vrow(self, x):
        return x.reshape((1, x.size))
    
    def logpdf_GMM(self, X, gmm):
        # Pre-allocate an array for storing log PDF values from all components
        S = np.zeros((len(gmm), X.shape[1]))

        # Calculate the density for each component using vectorization
        for idx, (w, mu, sigma) in enumerate(gmm):
            S[idx] = self.log_gau_pdf(X, mu, sigma) + np.log(w)
        S = np.vstack(S)
        return scipy.special.logsumexp(S, axis=0)
    
    def smooth_psi(self, sigma):
        U, s, Vt = np.linalg.svd(sigma)
        s[s < self.psiEig] = self.psiEig
        sigma_new = U @ (s.reshape(-1, 1) * U.T)
        return sigma_new
    
    def train_GMM_EM_Iteration(self, X, gmm):
        
        # E-step
        S = np.zeros((len(gmm), X.shape[1]))
        for idx, (w, mu, sigma) in enumerate(gmm):
            S[idx] = self.log_gau_pdf(X, mu, sigma) + np.log(w)
        
        S_normalized = np.exp(S - scipy.special.logsumexp(S, axis=0))
        
        # M-step
        gmm_new = []
        for idx in range(len(gmm)):
            gamma = S_normalized[idx]
            Z = gamma.sum()
            # Exploit broadcasting to compute the sum
            F = self.vcol((self.vrow(gamma) * X).sum(1))
            S = (self.vrow(gamma) * X) @ X.T
            mu_new = F / Z
            sigma_new = S / Z - mu_new @ mu_new.T
            w_new = Z / X.shape[1]
            
            if self.covariance_type == 'diagonal':
                sigma_new = sigma_new * np.eye(X.shape[0])
            
            gmm_new.append((w_new, mu_new, sigma_new))
        
        if self.covariance_type == 'tied':
            sigma_Ties = 0
            for w, mu, sigma in gmm_new:
                sigma_Ties += w * sigma
            gmm_new = [(w, mu, sigma_Ties) for w, mu, sigma in gmm_new]
        
        if self.psiEig is not None:
            gmm_new = [(w, mu, self.smooth_psi(sigma)) for w, mu, sigma in gmm_new]
        
        return gmm_new
        
    def train_GMM_EM(self, X, gmm):
        ll_old = self.logpdf_GMM(X, gmm).mean()
        ll_delta = None
        it = 1
        while ll_delta is None or ll_delta > self.tol:
            gmm_new = self.train_GMM_EM_Iteration(X, gmm)
            ll_new = self.logpdf_GMM(X, gmm_new).mean()
            ll_delta = np.abs(ll_new - ll_old)
            
            if self.verbose:
                print(f"Average ll: {ll_new:.8e}, Delta: {ll_delta:.8e} at iteration {it}")
            
            ll_old = ll_new
            gmm = gmm_new
            it += 1
        
        if self.verbose:
            print(f"Converged after {it} iterations, average ll: {ll_new:.8e}")
                
        return gmm
    
    def split_GMM_LBG(self, gmm):
        gmm_new = []
        if self.verbose:
            print(f"LBG - from {len(gmm)} to {2*len(gmm)} components")
        
        for w, mu, sigma in gmm:
            U, s, Vt = np.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * self.lbgAlpha
            gmm_new.append((w/2, mu + d, sigma))
            gmm_new.append((w/2, mu - d, sigma))
            
        return gmm_new
    
    def train_GMM_LBG_EM(self, X, n_components):
        # Compute the initial GMM
        mu = np.mean(X, axis=1).reshape(-1, 1)
        sigma = ((X - mu) @ ((X - mu).T)) / float(X.shape[1])
        
        if self.covariance_type == 'diagonal':
            sigma = sigma * np.eye(X.shape[0])
        
        if self.psiEig is not None:
            gmm = [(1.0, mu, self.smooth_psi(sigma))]
        else:
            gmm = [(1.0, mu, sigma)]
        
        # Iterate until the desired number of components is reached
        while len(gmm) < n_components:
            if self.verbose:
                print(f"Average ll before split: {self.logpdf_GMM(X, gmm).mean():.8e}")
            gmm = self.split_GMM_LBG(gmm)
            if self.verbose:
                print(f"Average ll after split: {self.logpdf_GMM(X, gmm).mean():.8e}")
            gmm = self.train_GMM_EM(X, gmm)
        
        return gmm
    
    def fit(self, X, y, n_features=2, folder=None, test_only=False):
        folder_path = os.path.join("__results__", "gmm", self.covariance_type, folder) if folder else None
        
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
            # print(f"Folder {folder_path} created successfully.")
        
        self.gmm_list = []
        
        for i in range(n_features):
            gmm_file = os.path.join(folder_path, f"gmm_{i}.pkl") if folder_path else None
            
            if test_only and gmm_file and os.path.exists(gmm_file):
                with open(gmm_file, 'rb') as f:
                    gmm = pickle.load(f)
                # print(f"GMM {i} loaded successfully.")
            else:
                # Assume this function trains the GMM and returns the model
                gmm = self.train_GMM_LBG_EM(X[:, y == i], self.n_components)
                
                if gmm_file:
                    with open(gmm_file, 'wb') as f:
                        pickle.dump(gmm, f)
                    # print(f"GMM {i} saved successfully.")
            
            self.gmm_list.append(gmm)
    
    def score(self, X):
        score = []
        for gmm in self.gmm_list:
            score.append(self.logpdf_GMM(X, gmm))
        score = np.vstack(score)
        score += np.log(np.ones(3)/3).reshape(-1, 1)
        
        return score
    
    def score_binary(self, X):
        return self.logpdf_GMM(X, self.gmm_list[1]) - self.logpdf_GMM(X, self.gmm_list[0]) 

    def predict(self, X):
        score = self.score(X)
        return np.argmax(score, axis=0)