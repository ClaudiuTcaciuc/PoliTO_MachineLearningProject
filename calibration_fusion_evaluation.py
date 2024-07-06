import utils
import numpy as np

from graph import *
from bayesian_decision_evaluation import *
from models.multivariate_gaussian_classifier import *
from models.logistic_regression_classifier import *
from models.svm_classifier import *
from models.svm_kernel_classifier import *
from models.gmm_clf import *

def compare_models(models_performances):
    best_models = utils.find_best_configuration(models_performances)
    
    print('\nBest models without calibration\n')
    for model, minDCF in best_models.items():
        print(f"\t{model}: MinDCF: {minDCF:.4f}")

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
