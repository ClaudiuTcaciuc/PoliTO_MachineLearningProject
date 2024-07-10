import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pp
from bayesian_decision_evaluation import *
from models.logistic_regression_classifier import LogisticRegressionWeighted

def load_data():
    #TODO: make the path not hard coded
    path = './data/trainData.txt'
    
    data_matrix = np.loadtxt(path, delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt(path, delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

def load_validation_data():
    #TODO: make the path not hard coded
    path = './data/evalData.txt'
    
    data_matrix = np.loadtxt(path, delimiter=',', usecols=range(0, 6), dtype=np.float64)
    data_labels = np.loadtxt(path, delimiter=',', usecols=6, dtype=int)
    
    return data_matrix.T, data_labels

def split_data(data, label, perc=(2.0/3.0), seed=0):
    """ Split the data 2/3 for train and 1/3 for test """
    
    n_train = int(data.shape[1] * perc)
    np.random.seed(seed)
    index = np.random.permutation(data.shape[1])
    index_train = index[:n_train]
    index_test = index[n_train:]

    data_train = data[:, index_train]
    label_train = label[index_train]
    data_test = data[:, index_test]
    label_test = label[index_test]
    
    return data_train, label_train, data_test, label_test

# def compute_statistics(data):
#     """ Compute the mean, variance, std and covariance matrix of the data """
#     mu_class = np.mean(data, axis=1).reshape(-1, 1)
#     print(f'Empirical dataset mean\n{mu_class}')
    
#     # Centered data
#     centered_data = data - mu_class
#     print(f'Centered data shape\n{centered_data.shape}')
    
#     # Covariance matrix
#     cov_matrix = np.cov(centered_data)
#     print(f'Covariance matrix shape\n{cov_matrix.shape}')

#     # Variance
#     var = np.var(data, axis=1).reshape(-1, 1)
#     print(f'Variance\n {var}')
    
#     # std
#     std = np.std(data, axis=1).reshape(-1, 1)
#     print(f'Standard deviation\n{std}')
    
def log_gau_pdf(X, mu, sigma):
    
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

def compute_accuracy_model_pca_threshold(X_train, y_train, X_test, y_test, models, threshold):
    results = {model_name: [] for model_name in models.keys()}
    
    for i in reversed(range(X_train.shape[0])):
        X_train_pca, eig_v = pp.pca(data=X_train, n_features=i+1, required_eigen_vectors=True)
        X_test_pca = np.dot(eig_v.T, X_test)
        
        for model_name, model in models.items():
            model.fit(X_train_pca, y_train)
            accuracy = model.predict_binary(X_test_pca, y_test, threshold)
            results[model_name].append((i+1, accuracy))
    
    plt.figure(figsize=(10, 6))
    
    for model_name, acc in results.items():
        features, accuracies = zip(*acc)
        plt.plot(features, accuracies, marker='o', label=model_name)
        
    plt.xlabel('Number of PCA features')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy with Varying PCA Features')
    plt.legend()
    plt.grid(True)
    plt.gca()
    plt.show()
    
    return results

def print_prior_prob(pi, cost_fp, cost_fn):
    cost_matrix, prior_class_prob, threshold = binary_cost_matrix(pi, cost_fp, cost_fn)
    print(f"pi: {pi}, prior_class_prob: {prior_class_prob}, cost_fp: {cost_fp}, cost_fn: {cost_fn}, threshold: {threshold:.4f}")
    print(f"Cost matrix:")
    for i in range(cost_matrix.shape[0]):
        print(f"\t{cost_matrix[i]}")
    print()

def compute_statistics(llr, y_true, prior, C_fp=1, C_fn=1, unique_labels=None, roc=False, bayes=False):
    cost_matrix, prior_class_prob, threshold = binary_cost_matrix(prior)
    
    min_DCF, best_threshold = compute_minDCF(llr, y_true, prior, unique_labels)
    y_pred = np.where(llr >= threshold, 1, 0)
    y_pred_best = np.where(llr >= best_threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred, unique_labels)
    cm_best = confusion_matrix(y_true, y_pred_best, unique_labels)
    acc = accuracy(cm)
    DCF, _, _ = compute_DCF(cm, cost_matrix, prior_class_prob)
    DCF_norm, _, _ = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
    
    print(f"\tMinDCF: {min_DCF:.4f}, Normalized DCF: {DCF_norm:.4f}, Accuracy: {acc*100:.2f}%\n")
    
    if roc:
        plot_ROC_curve(llr, y_true, cost_matrix, prior_class_prob, unique_labels)
    if bayes:
        plot_bayes_error(llr, y_true, unique_labels)
    
    return min_DCF, DCF, DCF_norm, acc

def compute_accuracy_model_pca(X_train, y_train, X_test, y_test, models, pi):
    results = {model_name: [] for model_name in models.keys()}
    
    for i in reversed(range(X_train.shape[0])):
        X_train_pca, eig_v = pp.pca(data=X_train, n_features=i+1, required_eigen_vectors=True)
        X_test_pca = np.dot(eig_v.T, X_test)
        
        for model_name, model in models.items():
            print(f"\tModel: {model_name}, PCA features: {i+1}")
            model.fit(X_train_pca, y_train)
            llr = model.score_binary(X_test_pca, y_test)
            minDCF, _, _, acc = compute_statistics(llr, y_test, pi)
            results[model_name].append((i+1, acc, minDCF))
    
    plt.figure(figsize=(10, 6))
    
    for model_name, acc in results.items():
        features, accuracies, dcfs = zip(*acc)
        plt.plot(features, dcfs, marker='o', label=model_name)
        
    plt.xlabel('Number of PCA features')
    plt.ylabel('min DCFs')
    plt.title(f'Model min DCF values with Varying PCA Features for pi={pi}')
    plt.legend()
    plt.grid(True)
    plt.gca()
    plt.show()
    
    return results

def find_best_configuration(models_performances):
    best_models = {}
    models_type = set([model.split('_')[0] for model in models_performances.keys()])

    for model_type in models_type:
        minDCF = np.inf
        best_model = None
        for model, performance in models_performances.items():
            if model_type in model:
                if performance['minDCF'] < minDCF:
                    minDCF = performance['minDCF']
                    best_model = model
        best_models[best_model] = minDCF

    return best_models

def extract_fold(X, idx, k_fold=10):
    return np.hstack([X[jdx::k_fold] for jdx in range(k_fold) if jdx != idx]), X[idx::k_fold]

def calibrate_system(score_s1, labels, prior, k_fold=10):
    calibrated_scores = []
    calibrated_labels = []
    for i in range(k_fold):
        score_cal, score_val = extract_fold(score_s1, i)
        labels_cal, labels_val = extract_fold(labels, i)
        
        clf = LogisticRegressionWeighted(lambda_=0, pi=prior, n_T=np.sum(labels_cal==1), n_F=np.sum(labels_cal==0))
        # print(f"\t\tCalibration fold {i + 1}\n")
        clf.fit(score_cal.reshape(1, -1), labels_cal)
        calibrated_sval = clf.score(score_val.reshape(1, -1)) - np.log(prior / (1 - prior))
        calibrated_scores.append(calibrated_sval)
        calibrated_labels.append(labels_val)
    
    calibrated_scores = np.hstack(calibrated_scores)
    calibrated_labels = np.hstack(calibrated_labels)

    return calibrated_scores, calibrated_labels