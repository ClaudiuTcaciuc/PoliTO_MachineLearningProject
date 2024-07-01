import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def binary_cost_matrix(pi, cost_fp=1, cost_fn=1):
    """ Compute the cost matrix for a given prior probability
        pi: prior probability
        cost_fp: cost of false positive
        cost_fn: cost of false negative
    """
    cost_matrix = np.array([[0, cost_fn], [cost_fp, 0]])
    prior_class_prob = np.array([1 - pi, pi])
    threshold = -np.log((pi * cost_fn) / ((1 - pi) * cost_fp))
    return cost_matrix, prior_class_prob, threshold

def confusion_matrix(y_true, y_pred, unique_labels=None):
    """ Compute the confusion matrix
        y_true: true labels
        y_pred: predicted labels
        unique_labels: unique labels in the dataset
    """
    if unique_labels is None:
        unique_labels = np.unique(y_true)
    n = len(unique_labels)
    conf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            conf_matrix[i, j] = np.sum((y_true == unique_labels[i]) & (y_pred == unique_labels[j]))
    return conf_matrix

def plot_confusion_matrix(conf_matrix):
    """ Plot the confusion matrix
        conf_matrix: confusion matrix
    """
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def compute_DCF(conf_matrix, cost_matrix, prior_class_prob):
    """ Compute the detection cost function
        conf_matrix: confusion matrix
        cost_matrix: cost matrix
        prior_class_prob: prior class probability
    """
    
    # cm = [[TN, FN], [FP, TP]]
    
    # P_fn = FN / (TP + FN)
    # P_fp = FP / (TN + FP)
    
    P_fn = conf_matrix[1, 0] / np.sum(conf_matrix[1, :])
    P_fp = conf_matrix[0, 1] / np.sum(conf_matrix[0, :])
    
    DCF = prior_class_prob[1] * cost_matrix[0, 1] * P_fn + prior_class_prob[0] * cost_matrix[1, 0] * P_fp
    return DCF, P_fn, P_fp

def compute_DCF_normalized(conf_matrix, cost_matrix, prior_class_prob):
    """ Compute the normalized detection cost function
        conf_matrix: confusion matrix
        cost_matrix: cost matrix
        prior_class_prob: prior class probability
    """
    DCF, P_fn, P_fp = compute_DCF(conf_matrix, cost_matrix, prior_class_prob)
    DCF_dummy = min(prior_class_prob[1] * cost_matrix[0, 1], prior_class_prob[0] * cost_matrix[1, 0])
    
    return DCF / DCF_dummy, P_fn, P_fp

def compute_minDCF(scores, y_true, pi, unique_labels=None):
    """ Compute the minimum detection cost function
        scores: scores of the samples (shape: (n_samples,))
        y_true: true labels
        pi: prior probability
        unique_labels: unique labels in the dataset (default None in case y_true does not contain all labels)
    """
    
    if unique_labels is None:
        unique_labels = np.unique(y_true)
    
    cost_matrix, prior_class_prob, _ = binary_cost_matrix(pi)
    
    # Sorting scores and labels
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    minDCF = float('inf')
    best_threshold = None
    
    N = len(y_true)
    
    for i in range(N - 1):
        y_pred = np.where(scores > sorted_scores[i], unique_labels[1], unique_labels[0])
        cm = confusion_matrix(y_true, y_pred, unique_labels)
        DCF, _, _ = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
        if DCF < minDCF:
            minDCF = DCF
            best_threshold = sorted_scores[i]
            
    return minDCF, best_threshold
    
def accuracy(conf_matrix):
    """ Compute the accuracy
        conf_matrix: confusion matrix
    """
    return np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

def plot_ROC_curve(scores, y_true, cost_matrix, prior_class_prob, unique_labels=None):
    """ Plot the ROC curve
        scores: scores of the samples (shape: (n_samples,))
        y_true: true labels
        conf_matrix: confusion matrix
        cost_matrix: cost matrix
        prior_class_prob: prior class probability
        unique_labels: unique labels in the dataset (default None in case y_true does not contain all labels)
    """
    
    if unique_labels is None:
        unique_labels = np.unique(y_true)
    
    # Sorting scores and labels
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    N = len(y_true)
    
    TPR = np.zeros(N)
    FPR = np.zeros(N)
    
    for i in range(N):
        y_pred = np.where(scores > sorted_scores[i], unique_labels[1], unique_labels[0])
        cm = confusion_matrix(y_true, y_pred, unique_labels)
        _, P_fn, P_fp = compute_DCF(cm, cost_matrix, prior_class_prob)
        TPR[i] = 1 - P_fn
        FPR[i] = P_fp
        
    plt.figure()
    plt.plot(FPR, TPR, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    
def plot_bayes_error(scores, y_true, unique_labels):
    eff_prior_log_odds = np.linspace(-3, 3, 21)
    eff_prior = 1 / (1 + np.exp(-eff_prior_log_odds))
    normalized_DCF = []
    normalized_minDCF = []
    
    for pi in eff_prior:
        cost_matrix, prior_class_prob, threshold = binary_cost_matrix(pi)
        minDCF, _ = compute_minDCF(scores, y_true, pi, unique_labels)
        normalized_minDCF.append(minDCF)
        y_pred = np.where(scores > threshold, unique_labels[1], unique_labels[0])
        cm = confusion_matrix(y_true, y_pred, unique_labels)
        DCF, _, _ = compute_DCF_normalized(cm, cost_matrix, prior_class_prob)
        normalized_DCF.append(DCF)
    
    plt.figure()
    plt.plot(eff_prior_log_odds, normalized_DCF, label='Normalized DCF', color='red')
    plt.plot(eff_prior_log_odds, normalized_minDCF, label='Normalized minDCF', color='blue')
    plt.xlabel('Effective prior probability')
    plt.ylabel('Detection Cost Function')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.title('Bayes error')
    plt.legend()
    plt.show()