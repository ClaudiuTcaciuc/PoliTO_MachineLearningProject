import preprocessing as pp
import utils
import numpy as np
import matplotlib.pyplot as plt
import os

from graph import *
from bayesian_decision_evaluation import *
from models.multivariate_gaussian_classifier import *
from models.logistic_regression_classifier import *
from models.svm_classifier import *
from models.svm_kernel_classifier import *
from models.gmm_clf import *

def do_data_analysis():
    print('Data Analysis\n')
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    X_pca = pp.pca(data=X, n_features=6)
    X_train_lda, eigen_vector = pp.lda(data=X_train, label=y_train, n_features=1, required_eigen_vectors=True)
    X_test_lda = np.dot(eigen_vector.T, X_test)

    classes = {
        'Fake': 'blue',
        'Real': 'orange'
    }
    
    X_f12 = X[:2, :] # Select the first two features
    X_f34 = X[2:4, :] # Select the third and fourth features
    X_f56 = X[4:6, :] # Select the fifth and sixth features
    
    X_pca_f12 = X_pca[:2, :]
    X_pca_f34 = X_pca[2:4, :]
    X_pca_f56 = X_pca[4:6, :]
    
    # Plot the data in the original space
    print('Plot the data in the original space\n')
    plot_histogram(X, y, classes)
    plot_scatter(X_f12, y, classes, title='Feature 1 vs Feature 2', features=[1, 2])
    plot_scatter(X_f34, y, classes, title='Feature 3 vs Feature 4', features=[3, 4])
    plot_scatter(X_f56, y, classes, title='Feature 5 vs Feature 6', features=[5, 6])
    
    # Plot the PCA explained variance
    print('Plot the PCA explained variance\n')
    plot_pca_explained_variance(X)
    
    # Plot the data in the PCA space
    print('Plot the data in the PCA space\n')
    plot_histogram(X_pca, y, classes, title='PCA Histogram')
    plot_scatter(X_pca_f12, y, classes, title='PCA Feature 1 vs Feature 2', features=[1, 2])
    plot_scatter(X_pca_f34, y, classes, title='PCA Feature 3 vs Feature 4', features=[3, 4])
    plot_scatter(X_pca_f56, y, classes, title='PCA Feature 5 vs Feature 6', features=[5, 6])
    
    # Plot the data in the LDA space
    print('Plot the data in the LDA space\n')
    plot_lda_histogram(X, y, classes, title='LDA Histogram')
    plot_histogram(X_train_lda, y_train, classes, title='LDA Histogram Train')
    plot_histogram(X_test_lda, y_test, classes, title='LDA Histogram Test')
    
    # Use LDA to classify the data
    print('Use LDA to classify the data\n')
    threshold = (X_train_lda[0, y_train == 1].mean() + X_train_lda[0, y_train == 0].mean()) / 2

    y_pred = np.zeros(shape=y_test.shape, dtype=np.int32)
    y_pred = np.where(X_test_lda[0] >= threshold, 1, 0)

    print(f'\tLDA Accuracy: {(np.mean(y_pred == y_test))*100:.2f}% (threshold: {threshold:.2f})\n')
    
    # try different thresholds
    thresholds = np.linspace(-10, 10, 1000)
    accuracy = 0

    for i, threshold in enumerate(thresholds):
        y_pred = np.where(X_test_lda[0] >= threshold, 1, 0)
        acc = np.mean(y_pred == y_test)
        if i % 100 == 0:
            print(f"\t\tLDA Accuracy: {acc*100:.2f}% with threshold: {threshold:.2f}")
        if acc > accuracy:
            accuracy = acc
            best_threshold = threshold
    
    print(f'\n\tLDA Best accuracy: {accuracy*100:.2f}% with threshold: {best_threshold:.2f}\n')
    
    # Use PCA to preprocess the data and LDA to classify the data
    for i in reversed(range(X_train.shape[0])):
        X_train_pca, eigen_vector = pp.pca(data=X_train, n_features=i+1, required_eigen_vectors=True)
        X_test_pca = np.dot(eigen_vector.T, X_test)
        
        X_train_lda, eigen_vector = pp.lda(data=X_train_pca, label=y_train, n_features=1, required_eigen_vectors=True)
        X_test_lda = np.dot(eigen_vector.T, X_test_pca)
        
        threshold = (X_train_lda[0, y_train == 1].mean() + X_train_lda[0, y_train == 0].mean()) / 2
        y_pred = np.zeros(shape=y_test.shape, dtype=np.int32)
        y_pred = np.where(X_test_lda[0] >= threshold, 1, 0)
        
        accuracy = np.mean(y_pred == y_test)
        print(f'\tPCA Accuracy with {i+1} features and LDA: {(accuracy)*100:.2f}%')

def plot_distribution_density(X=None):
    if X is None:
        X, _ = utils.load_data()

    Xplot = np.linspace(-8, 12, 1000).reshape(-1, 1)

    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    for i in range(6):
        X_1 = X[i:i+1, :].T
        mean = np.mean(X_1, axis=0).reshape(-1, 1)
        cov = np.dot((X_1 - mean).T, (X_1 - mean)) / X_1.shape[0]
        
        ax = axs[i//3, i%3]
        ax.hist(X_1, bins=50, density=True, ec='black')
        ax.plot(Xplot.ravel(), np.exp(utils.log_gau_pdf(Xplot, mean, cov)))
        ax.set_title(f'Feature {i+1}')
    plt.tight_layout()
    plt.show()

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
    print('\nPlotting correlation matrix \n')
    plot_correlation_matrix(X_f14, y)
    
    X_train_f14, y_train_f14, X_test_f14, y_test_f14 = utils.split_data(X_f14, y)

    print('\nMultivariate Gaussian Classifier on X_train_f14')
    for model_name, model in models_dict.items():
        model.fit(X_train_f14, y_train)
        accuracy = model.predict_binary(X_test_f14, y_test_f14, threshold)
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
        
    print('\nMultivariate Gaussian Classifier on X_train_14 and PCA')
    res = utils.compute_accuracy_model_pca(X_train_f14, y_train, X_test_f14, y_test, models_dict, threshold)
    for model_name, acc in res.items():
        print(f'\t{model_name} - Best Accuracy: {max(acc, key=lambda x: x[1])}')
        
    print('\nMultivariate Gaussian Classifier on X_train and PCA')
    res = utils.compute_accuracy_model_pca(X_train, y_train, X_test, y_test, models_dict, threshold)
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
        
    print('\nROC and Bayes error plot for the application with the lowest minDCF in PCA DATA: MVG PCA 6\n')
    pi = 0.1
    
    X_train_pca, eig_v = pp.pca(data=X_train, n_features=6, required_eigen_vectors=True)
    X_test_pca = np.dot(eig_v.T, X_test)

    models = {
        'MVG': MultivariateGaussianClassifier(),
        'NB': NaiveBayesClassifier(),
        'TC': TiedCovarianceClassifier(),
    }

    for model_name, model in models.items():
        print(f"Model: {model_name}")
        model.set_prior(pi)
        model.fit(X_train_pca, y_train)
        llr = model.score_binary(X_test_pca, y)
        utils.compute_statistics(llr, y_test, pi, unique_labels=np.unique(y), roc=True, bayes=True)

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
    
def compare_models(models_performances):
    best_models = utils.find_best_configuration(models_performances)
    
    print('\nBest models without calibration\n')
    for model, minDCF in best_models.items():
        print(f"\t{model}: MinDCF: {minDCF:.4f}")
    

def test_configuration_for_best_models():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
    priors = [0.1, 0.5, 0.9]
    covariance_type = 'diagonal'
    n_best_components = 8
    psiEig = 0.01
    
    print('\nTest configuration for best models\n')
    print('\nGMM on RAW DATA for different prior probabilities\n')
    for prior in priors:
        print(f"\tPrior probability: {prior}")
        model = GMM(n_components=n_best_components, covariance_type=covariance_type, psiEig=psiEig)
        folder = f"gmm_{covariance_type}_{n_best_components}_components"
        model.fit(X_train, y_train, n_features=2, folder=folder, test_only=True)
        score = model.score_binary(X_test)
        utils.compute_statistics(score, y_test, prior=prior, unique_labels=[0, 1], roc=False, bayes=True)

    eps = 1.0
    gamma = 1e-1
    C = 100.0
    prior = 0.1
    
    print('\nSVM RBF Kernel on RAW DATA for different prior probabilities\n')
    for prior in priors:
        print(f"\tPrior probability: {prior}")
        model = SVMClassifierRBFKernel(C=C, gamma=gamma, eps=eps)
        folder = f"svm_rbf_raw_data_gamma_{gamma:.1e}_eps_{eps}_C_{C:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train, y_train, folder, True)
        llr = model.score(X_test)
        utils.compute_statistics(llr, y_test, prior=prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    l = 0.032
    pi = 0.1
    pEmp = np.sum(y_train == 1) / len(y_train)
    model = QuadraticExpansion(lambda_=l)
    print('\nLogistic Regression Quadratic Expansion on RAW DATA for different prior probabilities\n')
    for prior in priors:
        print(f"\tPrior probability: {prior}")
        X_train_quad = model.expand(X_train)
        X_test_quad = model.expand(X_test)
        folder = f"lr_quad_raw_data_pi_{pi:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
        model.fit(X_train_quad, y_train, folder=folder, test_only=True)
        llr = model.score(X_test_quad) - np.log(pEmp / (1 - pEmp))
        utils.compute_statistics(llr, y_test, pi, unique_labels=[0, 1], roc=False, bayes=True)

def calibration_fusion_evaluation_models():
    X, y = utils.load_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    X_eval, y_eval = utils.load_validation_data()
    k_fold = 10
    prior_calibration = 0.2
    
    
    test_only = True
    covariance_type = 'diagonal'
    n_best_components = 8
    prior = 0.1
    psiEig = 0.01

    model_gmm = GMM(n_components=n_best_components, covariance_type=covariance_type, psiEig=psiEig)
    folder = f"gmm_{covariance_type}_{n_best_components}_components"
    model_gmm.fit(X_train, y_train, n_features=2, folder=folder, test_only=test_only)
    gmm_score = model_gmm.score_binary(X_test)
    
    eps = 1.0
    gamma = 1e-1
    C = 100.0
    prior = 0.1

    model_svm = SVMClassifierRBFKernel(C=C, gamma=gamma, eps=eps)
    folder = f"svm_rbf_raw_data_gamma_{gamma:.1e}_eps_{eps}_C_{C:.1e}".replace('.', '_').replace('e-0', 'e-')
    model_svm.fit(X_train, y_train, folder, test_only)
    svm_score = model_svm.score(X_test)
    
    l = 0.032
    prior = 0.1
    n_T = np.sum(y_train == 1)
    n_F = np.sum(y_train == 0)
    pEmp = n_T / (n_T + n_F)

    model_qlr = QuadraticExpansion(lambda_=l)
    X_train_quad = model_qlr.expand(X_train)
    X_test_quad = model_qlr.expand(X_test)
    folder = f"lr_quad_raw_data_pi_{prior:.1e}_lambda_{l:.1e}".replace('.', '_').replace('e-0', 'e-')
    model_qlr.fit(X_train_quad, y_train, folder=folder, test_only=test_only)
    lr_score = model_qlr.score(X_test_quad) - np.log(pEmp / (1 - pEmp))
    
    gmm_calibrated_scores, gmm_calibrated_labels = utils.calibrate_system(gmm_score, y_test, prior_calibration)
    svm_calibrated_scores, svm_calibrated_labels = utils.calibrate_system(svm_score, y_test, prior_calibration)
    lr_calibrated_scores, lr_calibrated_labels = utils.calibrate_system(lr_score, y_test, prior_calibration)

    clf_cal_gmm = LogisticRegressionWeighted(lambda_=0, pi=prior_calibration, n_T=np.sum(gmm_calibrated_labels==1), n_F=np.sum(gmm_calibrated_labels==0))
    clf_cal_svm = LogisticRegressionWeighted(lambda_=0, pi=prior_calibration, n_T=np.sum(svm_calibrated_labels==1), n_F=np.sum(svm_calibrated_labels==0))
    clf_cal_lr = LogisticRegressionWeighted(lambda_=0, pi=prior_calibration, n_T=np.sum(lr_calibrated_labels==1), n_F=np.sum(lr_calibrated_labels==0))

    clf_cal_gmm.fit(gmm_score.reshape(1, -1), y_test)
    clf_cal_svm.fit(svm_score.reshape(1, -1), y_test)
    clf_cal_lr.fit(lr_score.reshape(1, -1), y_test)
    
    print(f"GMM before calibration with prior {prior}")
    utils.compute_statistics(gmm_score, y_test, prior, unique_labels=[0, 1], roc=False, bayes=True)

    print(f"GMM calibrated with prior {prior}")
    utils.compute_statistics(gmm_calibrated_scores, gmm_calibrated_labels, prior, unique_labels=[0, 1], roc=False, bayes=True)

    print(f"SVM before calibration with prior {prior}")
    utils.compute_statistics(svm_score, y_test, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"SVM calibrated with prior {prior}")
    utils.compute_statistics(svm_calibrated_scores, svm_calibrated_labels, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"LR before calibration with prior {prior}")
    utils.compute_statistics(lr_score, y_test, prior, unique_labels=[0, 1], roc=False, bayes=True)

    print(f"LR calibrated with prior {prior}")
    utils.compute_statistics(lr_calibrated_scores, lr_calibrated_labels, prior, unique_labels=[0, 1], roc=False, bayes=True)

    fused_scores = []
    fused_labels = []

    for idx in range(k_fold):
        gmm_cal, gmm_val = utils.extract_fold(gmm_score, idx)
        svm_cal, svm_val = utils.extract_fold(svm_score, idx)
        lr_cal, lr_val = utils.extract_fold(lr_score, idx)
        labels_cal, labels_val = utils.extract_fold(y_test, idx)
        
        score_cal = np.vstack([gmm_cal, svm_cal, lr_cal])
        score_val = np.vstack([gmm_val, svm_val, lr_val])
        
        clf_fusion = LogisticRegressionWeighted(lambda_=0, pi=prior_calibration, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
        clf_fusion.fit(score_cal, labels_cal)
        fused_score = clf_fusion.score(score_val) - np.log(prior_calibration / (1 - prior_calibration))
        fused_scores.append(fused_score)
        fused_labels.append(labels_val)

    fused_scores = np.hstack(fused_scores)
    fused_labels = np.hstack(fused_labels)

    print(f"Fusion with prior {prior}")
    utils.compute_statistics(fused_scores, fused_labels, prior, unique_labels=[0, 1], roc=False, bayes=True)

    print(f"EVALUATION ON GMM 8 DIAG\n")
    gmm_score_eval = model_gmm.score_binary(X_eval)
    gmm_score_eval_calibrated = clf_cal_gmm.score(gmm_score_eval.reshape(1, -1)) - np.log(prior_calibration / (1 - prior_calibration))

    print(f"GMM evaluation set with prior {prior}")
    utils.compute_statistics(gmm_score_eval, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"GMM evaluation set calibrated with prior {prior}")
    utils.compute_statistics(gmm_score_eval_calibrated, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"EVALUATION ON SVM RBF\n")
    svm_score_eval = model_svm.score(X_eval)
    svm_score_eval_calibrated = clf_cal_svm.score(svm_score_eval.reshape(1, -1)) - np.log(prior_calibration / (1 - prior_calibration))
    
    print(f"SVM evaluation set with prior {prior}")
    utils.compute_statistics(svm_score_eval, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"SVM evaluation set calibrated with prior {prior}")
    utils.compute_statistics(svm_score_eval_calibrated, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    X_eval_quad = model_qlr.expand(X_eval)
    lr_score_eval = model_qlr.score(X_eval_quad) - np.log(pEmp / (1 - pEmp))
    lr_score_eval_calibrated = clf_cal_lr.score(lr_score_eval.reshape(1, -1)) - np.log(prior_calibration / (1 - prior_calibration))
    
    print(f"LR evaluation set with prior {prior}")
    utils.compute_statistics(lr_score_eval, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"LR evaluation set calibrated with prior {prior}")
    utils.compute_statistics(lr_score_eval_calibrated, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)
    
    print(f"Fusion evaluation set with prior {prior}")
    score_matrix = np.vstack([gmm_score, svm_score, lr_score])
    clf_fusion = LogisticRegressionWeighted(lambda_=0, pi=prior_calibration, n_T=np.sum(y_test==1), n_F=np.sum(y_test==0))
    clf_fusion.fit(score_matrix, y_test)

    score_eval_matrix = np.vstack([gmm_score_eval, svm_score_eval, lr_score_eval])
    fused_score_eval = clf_fusion.score(score_eval_matrix) - np.log(prior_calibration / (1 - prior_calibration))
    
    print(f"Fusion evaluation set with prior {prior}")
    utils.compute_statistics(fused_score_eval, y_eval, prior, unique_labels=[0, 1], roc=False, bayes=True)

def train_and_test_gmm_classifier_bayesian_on_evaluation_set():
    X, y = utils.load_data()
    X_eval, y_eval = utils.load_validation_data()
    X_train, y_train, X_test, y_test = utils.split_data(X, y)
    
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
    
def main():
    # do_data_analysis()
    # plot_distribution_density()
    # test_multivariate_gaussian_classifier_base()
    # train_and_test_multivariate_gaussian_classifier_bayesian()
    # model_performance_lr = train_and_test_logistic_regression_bayesian()
    # model_performance_svm = train_and_test_support_vector_machines_bayesian()
    # model_performance_gmm = train_and_test_gmm_classifier_bayesian()
    
    # model_performance = {**model_performance_lr, **model_performance_svm, **model_performance_gmm}
    # compare_models(model_performance)
    # test_configuration_for_best_models()
    calibration_fusion_evaluation_models()
    model_performance_gmm_eval = train_and_test_gmm_classifier_bayesian_on_evaluation_set()
    compare_models(model_performance_gmm_eval)

if __name__ == '__main__':
    main()