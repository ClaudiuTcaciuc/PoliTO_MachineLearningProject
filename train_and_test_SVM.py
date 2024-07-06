import preprocessing as pp
import utils
import numpy as np
import os

from graph import *
from bayesian_decision_evaluation import *
from models.svm_classifier import *
from models.svm_kernel_classifier import *

def train_and_test_support_vector_machines_bayesian():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    test_only = True if os.path.join("__results__", "svm") else False
    models_performances = {}
    K = 1.0
    C = np.logspace(-5, 0, 11)
    pi = 0.1
    
    print('\nSupport Vector Machine Classifier Linear Version on RAW DATA for different C values\n')

    min_DCF_list_linear = []
    DCF_norm_list_linear = []

    for c in C:
        print(f"C: {c:.2e}")
        model = SVMClassifier(C=c, K=K)
        folder = f"svm_linear_raw_data_K_{K:.1e}_C_{c:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train, y_train, folder, test_only)
        llr = model.score(X_test)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_linear.append(min_DCF)
        DCF_norm_list_linear.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_linear, DCF_norm_list_linear, C, title='SVM Linear RAW DATA')
    
    print('\nSupport Vector Machine Classifier Linear Version on STD DATA for different C values\n')
    X_train_std, mean, std = pp.standardize(X_train, return_params=True)
    X_test_std = X_test - mean / std
    
    min_DCF_list_linear_std = []
    DCF_norm_list_linear_std = []

    for c in C:
        print(f"C: {c:.2e}")
        model = SVMClassifier(C=c, K=K)
        folder = f"svm_linear_std_data_K_{K:.1e}_C_{c:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train_std, y_train, folder, test_only)
        llr = model.score(X_test_std)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_linear_std.append(min_DCF)
        DCF_norm_list_linear_std.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_linear_std, DCF_norm_list_linear_std, C, title='SVM Linear STD DATA')
    
    print('\nSupport Vector Machine Classifier Poly Kernel Version on RAW DATA for different C values\n')
    d = 2
    c = 1
    eps = 0
    
    min_DCF_list_poly = []
    DCF_norm_list_poly = []

    for ci in C:
        print(f"C: {ci:.2e}")
        model = SVMClassifierPolyKernel(C=ci, eps=eps, degree=d, delta=c)
        folder = f"svm_poly_raw_data_degree_{d}_delta_{c}_eps_{eps}_C_{ci:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train, y_train, folder, test_only)
        llr = model.score(X_test)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_poly.append(min_DCF)
        DCF_norm_list_poly.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_poly, DCF_norm_list_poly, C, title='SVM Poly RAW DATA')

    print('\nSupport Vector Machine Classifier RBF Kernel Version on RAW DATA for different C values\n')
    eps = 1.0
    gamma = [1e-4, 1e-3, 1e-2, 1e-1]
    C = np.logspace(-3, 2, 11)
    
    min_DCF_list_rbf = {g: [] for g in gamma}
    DCF_norm_list_rbf = {g: [] for g in gamma}

    # all the possible combinations of gamma and C
    for g in gamma:
        for c in C:
            print(f"Gamma: {g}, C: {c:.2e}")
            model = SVMClassifierRBFKernel(C=c, gamma=g, eps=eps)
            folder = f"svm_rbf_raw_data_gamma_{g:.1e}_eps_{eps}_C_{c:.1e}".replace('.', '_').replace('e-0', 'e-')
            model.fit(X_train, y_train, folder, test_only)
            llr = model.score(X_test)
            min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
            min_DCF_list_rbf[g].append(min_DCF)
            DCF_norm_list_rbf[g].append(DCF_norm)
            models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
        
    plot_overall_DCF_rbf(min_DCF_list_rbf, DCF_norm_list_rbf, C, gamma)

    return models_performances
