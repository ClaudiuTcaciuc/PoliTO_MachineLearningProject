import utils
import os

from graph import *
from bayesian_decision_evaluation import *
from models.gmm_clf import *

def train_and_test_gmm_classifier_bayesian():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    test_only = True if os.path.join("__results__", "gmm") else False
    
    models_performances = {}
    n_components = [1, 2, 4, 8, 16, 32]
    psiEig = 0.01
    pi = 0.1
    
    print('\nGMM Full Covariance Classifier on RAW DATA for different number of components\n')
    covariance_type = 'full'

    min_DCF_list_full = []
    DCF_norm_list_full = []

    for n in n_components:
        print(f"Number of components: {n}")
        model = GMM(n_components=n, covariance_type=covariance_type, psiEig=psiEig)
        folder = f"gmm_{covariance_type}_{n}_components"
        model.fit(X_train, y_train, n_features=2, folder=folder, test_only=test_only)
        score = model.score_binary(X_test)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(score, y_test, prior=pi, unique_labels=[0, 1])
        min_DCF_list_full.append(min_DCF)
        DCF_norm_list_full.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
        
    print('\nGMM Diagonal Covariance Classifier on RAW DATA for different number of components\n')
    covariance_type = 'diagonal'

    min_DCF_list_full = []
    DCF_norm_list_full = []

    for n in n_components:
        print(f"Number of components: {n}")
        model = GMM(n_components=n, covariance_type=covariance_type, psiEig=psiEig)
        folder = f"gmm_{covariance_type}_{n}_components"
        model.fit(X_train, y_train, n_features=2, folder=folder, test_only=test_only)
        score = model.score_binary(X_test)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(score, y_test, prior=pi, unique_labels=[0, 1])
        min_DCF_list_full.append(min_DCF)
        DCF_norm_list_full.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}

    return models_performances

def train_and_test_gmm_classifier_bayesian_on_evaluation_set():
    X, y = utils.load_data()
    X_eval, y_eval = utils.load_validation_data()
    X_train, y_train, _, _ = utils.split_data(X, y)
    
    test_only = True if os.path.join("__results__", "gmm") else False
    
    models_performances = {}
    n_components = [1, 2, 4, 8, 16, 32]
    psiEig = 0.01
    pi = 0.1
    
    print('\nGMM Full Covariance Classifier on RAW DATA for different number of components on Evaluation set\n')
    covariance_type = 'full'

    min_DCF_list_full = []
    DCF_norm_list_full = []

    for n in n_components:
        print(f"Number of components: {n}")
        model = GMM(n_components=n, covariance_type=covariance_type, psiEig=psiEig)
        folder = f"gmm_{covariance_type}_{n}_components"
        model.fit(X_train, y_train, n_features=2, folder=folder, test_only=test_only)
        score = model.score_binary(X_eval)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(score, y_eval, prior=pi, unique_labels=[0, 1])
        min_DCF_list_full.append(min_DCF)
        DCF_norm_list_full.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
        
    print('\nGMM Diagonal Covariance Classifier on RAW DATA for different number of components on Evaluation set\n')
    covariance_type = 'diagonal'

    min_DCF_list_full = []
    DCF_norm_list_full = []

    for n in n_components:
        print(f"Number of components: {n}")
        model = GMM(n_components=n, covariance_type=covariance_type, psiEig=psiEig)
        folder = f"gmm_{covariance_type}_{n}_components"
        model.fit(X_train, y_train, n_features=2, folder=folder, test_only=test_only)
        score = model.score_binary(X_eval)
        min_DCF, _, DCF_norm, _ = utils.compute_statistics(score, y_eval, prior=pi, unique_labels=[0, 1])
        min_DCF_list_full.append(min_DCF)
        DCF_norm_list_full.append(DCF_norm)
        models_performances[folder] = {'minDCF': min_DCF, 'DCF_norm': DCF_norm}
    
    print(f"GMM Results on Evaluation set\n")
    for model, performance in models_performances.items():
        print(f"Model: {model}")
        for key, value in performance.items():
            print(f"\t{key}: {value:.4f}")
    
    return models_performances
