import preprocessing as pp
import utils
import numpy as np
import os

from graph import *
from bayesian_decision_evaluation import *
from models.logistic_regression_classifier import *

def train_and_test_logistic_regression_bayesian():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    X_train_reduced = X_train[:, ::50]
    y_train_reduced = y_train[::50]
    
    models_performances = {}
    
    lambda_ = np.logspace(-4, 2, 13)
    pi = 0.1
    n_T = np.sum(y_train == 1)
    n_F = np.sum(y_train == 0)
    pEmp = n_T / (n_T + n_F)
    
    test_only = True if os.path.join("__results__", "logistic_regression") else False
    
    print('\nLogistic Regression Classifier on RAW DATA for different lambda values\n')
    min_DCF_list_base = []
    DCF_norm_list_base = []

    for l in lambda_:
        print(f"Lambda: {l:.2e}")
        model = LogisticRegression(lambda_=l)
        folder = f"lr_base_raw_data_pi_{pi:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train, y_train, folder=folder, test_only=test_only)
        llr = model.score(X_test) - np.log(pEmp / (1 - pEmp))
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_base.append(min_DCF)
        DCF_norm_list_base.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_base, DCF_norm_list_base, lambda_, title='LR RAW DATA')
    
    print('\nLogistic Regression Classifier on REDUCED RAW DATA for different lambda values\n')
    min_DCF_list_base_reduced = []
    DCF_norm_list_base_reduced = []

    for l in lambda_:
        print(f"Lambda: {l:.2e}")
        model = LogisticRegression(lambda_=l)
        model.fit(X_train_reduced, y_train_reduced, test_only=test_only)
        llr = model.score(X_test) - np.log(pEmp / (1 - pEmp))
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_base_reduced.append(min_DCF)
        DCF_norm_list_base_reduced.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_base_reduced, DCF_norm_list_base_reduced, lambda_, title='LR REDUCED RAW DATA')
    
    print('\nLogistic Regression Weighted Classifier on RAW DATA for different lambda values\n')
    min_DCF_list_weight = []
    DCF_norm_list_weight = []

    for l in lambda_:
        print(f"Lambda: {l:.2e}")
        model = LogisticRegressionWeighted(lambda_=l, pi=pi, n_T=n_T, n_F=n_F)
        folder = f"lr_weighted_raw_data_pi_{pi:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train, y_train, folder=folder, test_only=test_only)
        llr = model.score(X_test) - np.log(pi / (1 - pi))
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_weight.append(min_DCF)
        DCF_norm_list_weight.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}

    plot_overall_DCF(min_DCF_list_weight, DCF_norm_list_weight, lambda_, title='LRW RAW DATA')
    
    print('\nQuadratic Logistic Regression Classifier on RAW DATA for different lambda values\n')
    
    min_DCF_list_quad = []
    DCF_norm_list_quad = []

    for l in lambda_:
        print(f"Lambda: {l:.2e}")
        model = QuadraticExpansion(lambda_=l)
        X_train_quad = model.expand(X_train)
        X_test_quad = model.expand(X_test)
        folder = f"lr_quad_raw_data_pi_{pi:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train_quad, y_train, folder=folder, test_only=test_only)
        llr = model.score(X_test_quad) - np.log(pEmp / (1 - pEmp))
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_quad.append(min_DCF)
        DCF_norm_list_quad.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_quad, DCF_norm_list_quad, lambda_, title='LRQ RAW DATA')
    
    print('\nLogistic Regression Classifier on STD DATA for different lambda values\n')
    X_train_std, mean, std = pp.standardize(X_train, return_params=True)
    X_test_std = X_test - mean / std
    
    min_DCF_list_base_std = []
    DCF_norm_list_base_std = []

    for l in lambda_:
        print(f"Lambda: {l:.2e}")
        model = LogisticRegression(lambda_=l)
        folder = f"lr_base_std_data_pi_{pi:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train_std, y_train, folder=folder, test_only=test_only)
        llr = model.score(X_test_std) - np.log(pEmp / (1 - pEmp))
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_base_std.append(min_DCF)
        DCF_norm_list_base_std.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_base_std, DCF_norm_list_base_std, lambda_, title='LR STD DATA')

    print('\nLogistic Regression Weighted Classifier on RAW DATA for different lambda values\n')
    
    min_DCF_list_weight_std = []
    DCF_norm_list_weight_std = []

    for l in lambda_:
        print(f"Lambda: {l:.2e}")
        model = LogisticRegressionWeighted(lambda_=l, pi=pi, n_T=n_T, n_F=n_F)
        folder = f"lr_weighted_std_data_pi_{pi:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train_std, y_train, folder=folder, test_only=test_only)
        llr = model.score(X_test_std) - np.log(pi / (1 - pi))
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(llr, y_test, pi)
        min_DCF_list_weight_std.append(min_DCF)
        DCF_norm_list_weight_std.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    plot_overall_DCF(min_DCF_list_weight_std, DCF_norm_list_weight_std, lambda_, title='LRW STD DATA')
    
    return models_performances
    