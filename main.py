from data_analysis import *
from train_and_test_MVG import *
from train_and_test_LR import *
from train_and_test_SVM import *
from train_and_test_GMM import *
from calibration_fusion_evaluation import *

def main():
    do_data_analysis()
    test_multivariate_gaussian_classifier_base()
    train_and_test_multivariate_gaussian_classifier_bayesian()
    model_performance_lr = train_and_test_logistic_regression_bayesian()
    model_performance_svm = train_and_test_support_vector_machines_bayesian()
    model_performance_gmm = train_and_test_gmm_classifier_bayesian()
    
    model_performance = {**model_performance_lr, **model_performance_svm, **model_performance_gmm}
    compare_models(model_performance)
    test_configuration_for_best_models()
    calibration_fusion_evaluation_models()
    model_performance_gmm_eval = train_and_test_gmm_classifier_bayesian_on_evaluation_set()
    compare_models(model_performance_gmm_eval)
    pass

if __name__ == '__main__':
    main()