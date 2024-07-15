import preprocessing as pp
import utils
import numpy as np

from graph import *
from models.multivariate_gaussian_classifier import *

def test_multivariate_gaussian_classifier_base():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    X_f14 = X[:-2, :]
    X_train_12 = X_train[:2, :]
    X_test_12 = X_test[:2, :]
    X_train_34 = X_train[2:4, :]
    X_test_34 = X_test[2:4, :]
    
    pi = 0.5
    
    # define models
    models_dict = {
        'MVG': MultivariateGaussianClassifier(),
        'NB': NaiveBayesClassifier(),
        'TC': TiedCovarianceClassifier(),
    }
    # set threshold for binary classification
    threshold = -np.log(pi / (1 - pi))

    print('\nMultivariate Gaussian Classifier on X_train')
    for model_name, model in models_dict.items():
        model.fit(X_train, y_train)
        accuracy = model.predict_binary(X_test, y_test, threshold)
        print(f'\t{model_name} Accuracy: {accuracy*100:.2f}% (threshold: {threshold:.2f})')
    
    print('\nPlotting correlation matrix \n')
    plot_correlation_matrix(X, y)
    print('\nPlotting correlation matrix on features 1 to 4\n')
    plot_correlation_matrix(X_f14, y)
    
    X_train_f14, _, X_test_f14, _ = utils.split_data(X_f14, y)

    print('\nMultivariate Gaussian Classifier on X_train_f14')
    for model_name, model in models_dict.items():
        model.fit(X_train_f14, y_train)
        accuracy = model.predict_binary(X_test_f14, y_test, threshold)
        print(f'\t{model_name} Accuracy: {accuracy*100:.2f}% (threshold: {threshold:.2f})')
        
    print('\nMultivariate Gaussian Classifier on X_train_12')
    for model_name, model in models_dict.items():
        model.fit(X_train_12, y_train)
        accuracy = model.predict_binary(X_test_12, y_test, threshold)
        print(f'\t{model_name} Accuracy: {accuracy*100:.2f}% (threshold: {threshold:.2f})')
        
    print('\nMultivariate Gaussian Classifier on X_train_34')
    for model_name, model in models_dict.items():
        model.fit(X_train_34, y_train)
        accuracy = model.predict_binary(X_test_34, y_test, threshold)
        print(f'\t{model_name} Accuracy: {accuracy*100:.2f}% (threshold: {threshold:.2f})')
        
    print('\nMultivariate Gaussian Classifier on X_train and PCA')
    res = utils.compute_accuracy_model_pca_threshold(X_train, y_train, X_test, y_test, models_dict, threshold=threshold)
    for model_name, acc in res.items():
        print(f'\t{model_name} - Best Accuracy: {max(acc, key=lambda x: x[1])}')
        
    print('\nMultivariate Gaussian Classifier on X_train_14 and PCA')
    res = utils.compute_accuracy_model_pca_threshold(X_train_f14, y_train, X_test_f14, y_test, models_dict, threshold=threshold)
    for model_name, acc in res.items():
        print(f'\t{model_name} - Best Accuracy: {max(acc, key=lambda x: x[1])}')


def train_and_test_multivariate_gaussian_classifier_bayesian():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    pie = [0.1, 0.5, 0.9]
    Applications = [[0.5, 1.0, 1.0], [0.9, 1.0, 1.0], [0.1, 1.0, 1.0], [0.5, 1.0, 9.0], [0.5, 9.0, 1.0]]
    
    models_dict = {
        'MVG': MultivariateGaussianClassifier(),
        'NB': NaiveBayesClassifier(),
        'TC': TiedCovarianceClassifier(),
    }
    print('\nPrior probability for different applications\n')
    utils.print_prior_prob(0.5, 1, 1)
    utils.print_prior_prob(0.9, 1, 1)
    utils.print_prior_prob(0.1, 1, 1)
    utils.print_prior_prob(0.5, 1, 9)
    utils.print_prior_prob(0.5, 9, 1)
    
    print('\nMultivariate Gaussian Classifier on RAW DATA for different applications\n')
    for pi, C_fp, C_fn in Applications:
        print(f"Prior probability: {pi}, C_fp: {C_fp}, C_fn: {C_fn}")
        for model_name, model in models_dict.items():
            print(f"\tModel: {model_name}")
            model.set_prior(pi)
            model.fit(X_train, y_train)
            llr = model.score_binary(X_test, y)
            utils.compute_statistics(llr, y_test, pi, C_fp, C_fn, unique_labels=np.unique(y))
    
    print('\nMultivariate Gaussian Classifier on RAW DATA for different prior probabilities\n')
    for pi in pie:
        print(f"Prior probability: {pi}")
        for model_name, model in models_dict.items():
            print(f"\tModel: {model_name}")
            model.set_prior(pi)
            model.fit(X_train, y_train)
            llr = model.score_binary(X_test, y)
            utils.compute_statistics(llr, y_test, pi, unique_labels=np.unique(y))
            
    print('\nMultivariate Gaussian Classifier on PCA DATA for different applications\n')
    
    for pi in pie:
        res = utils.compute_accuracy_model_pca(X_train, y_train, X_test, y_test, models_dict, pi)
        for model_name, acc in res.items():  # for clean print
            print(f'\t{model_name} - Best Accuracy: {max(acc, key=lambda x: x[1])}')
        for model_name, acc in res.items():
            print(f'\t{model_name} - Best minDCF: {min(acc, key=lambda x: x[2])}')
        
    print('\nROC and Bayes error plot for the application with the lowest minDCF in PCA DATA: MVG PCA 5\n')
    pi = 0.1
    
    X_train_pca, eig_v = pp.pca(data=X_train, n_features=5, required_eigen_vectors=True)
    X_test_pca = np.dot(eig_v.T, X_test)

    for model_name, model in models_dict.items():
        print(f"Model: {model_name}")
        model.set_prior(pi)
        model.fit(X_train_pca, y_train)
        llr = model.score_binary(X_test_pca, y)
        utils.compute_statistics(llr, y_test, pi, unique_labels=np.unique(y), roc=True, bayes=True)
