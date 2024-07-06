import preprocessing as pp
import utils
import numpy as np
from graph import *

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
    
    print('\n Plot the data distribution density on the RAW space\n')
    plot_distribution_density()

