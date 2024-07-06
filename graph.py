import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import scipy
import utils
import preprocessing as pre

def plot_histogram(data: np.array, label:np.array, classes: dict[str, str], title:str=""):
    """ Plot histograms for dataset features
    
        Parameters:
            data: n_features, n_samples
            label: n_samples
            classes: dictionary with class names and colors
    """
    
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    plt.figure()
    n_rows, _ = data.shape
    
    if n_rows > 1:
        for i in range(n_rows):
            plt.subplot(len(classes), n_rows // len(classes), i+1)
            for j, kv in enumerate(classes.items()):
                class_name, class_color = kv
                plt.hist(data[i, label == j], alpha=0.5, density=True, label=class_name, ec=class_color)
            plt.title(f'Feature {i+1}')
            plt.legend(loc='best')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    else:
        for i, kv in enumerate(classes.items()):
            class_name, class_color = kv
            plt.hist(data[0, label == i], alpha=0.5, density=True, label=class_name, ec=class_color)
        plt.title(title)
        plt.legend(loc='best')
        plt.show()

def plot_scatter(data: np.array, label: np.array, classes: dict[str, str], title:str="", features:list=None):
    """ Plot scatter plots for dataset features
    
        Parameters:
            data: n_features, n_samples
            label: n_samples
            classes: dictionary with class names and colors
    """
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    plt.figure(figsize=(7, 7))
    
    n_rows, _ = data.shape
    
    for i in range(n_rows):
        for j in range(i, n_rows):
            plt.subplot(n_rows, n_rows, i*n_rows + j + 1)
            if i != j:
                for k, v in enumerate(classes.items()):
                    class_name, class_color = v
                    plt.scatter(data[j, label == k], data[i, label == k], label=class_name, s=2, c=class_color)
                plt.xlabel(f'Feature {j+1 if features is None else features[j]}')
                plt.ylabel(f'Feature {i+1 if features is None else features[i]}')
                plt.legend(loc='best')
            else:
                for k, v in enumerate(classes.items()):
                    class_name, class_color = v
                    plt.hist(data[i, label == k], alpha=0.5, label=class_name, density=True, ec=class_color)
                plt.legend(loc='best')
                plt.ylabel(f'Feature {i+1 if features is None else features[i]}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data: np.array, label: np.array):
    """
    Plot correlation matrix for dataset features.
    
    Parameters:
        data: np.array, shape (n_features, n_samples)
            Input dataset.
        label: np.array, shape (n_samples,)
            Labels for dataset samples.
    """
    
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    correlation_matrix = np.corrcoef(data)
    class_corr = {}
    
    for class_label in np.unique(label):
        class_corr[class_label] = np.corrcoef(data[:, label == class_label])
        
    num_classes = len(class_corr)
    num_plots = num_classes + 1
    _, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
                xticklabels=np.arange(1, len(correlation_matrix) + 1),
                yticklabels=np.arange(1, len(correlation_matrix) + 1),
                ax=axes[0])
    axes[0].set_title('Overall Correlation Matrix')

    for i, class_label in enumerate(class_corr.keys(), start=1):
        sns.heatmap(class_corr[class_label], annot=True, cmap='coolwarm', fmt=".2f", square=True,
                    xticklabels=np.arange(1, len(correlation_matrix) + 1),
                    yticklabels=np.arange(1, len(correlation_matrix) + 1),
                    ax=axes[i])
        axes[i].set_title(f'Class {class_label} Correlation Matrix')

    plt.tight_layout()
    plt.show()

def plot_pca_explained_variance(data:np.array):
    """ Plot the explained variance for each principal component
    
        Parameters:
            data: n_features, n_samples
    """
    
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    mean = np.mean(data, axis=1).reshape(-1, 1)
    centered_data = data - mean
    cov = np.dot(centered_data, centered_data.T) / data.shape[1]
    
    eigen_values, _ = scipy.linalg.eigh(cov)
    
    total_eigen_values = np.sum(eigen_values)
    var_exp = [(i / total_eigen_values) for i in sorted(eigen_values, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(0, len(var_exp)), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(0, len(cum_var_exp)), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance ratio')
    plt.legend(loc='best')
    plt.show()

def plot_lda_histogram(data: np.array, label: np.array, classes: dict[str, str], title:str=""):
    """ Plot histograms for dataset features
    
        Parameters:
            data: n_features, n_samples
            label: n_samples
            classes: dictionary with class names and colors
    """
    
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    lda_data = pre.lda(data=data, label=label, n_features=1)
    plot_histogram(data=lda_data, label=label, classes=classes, title=title)

def plot_graph():
    data, label = utils.load_data()
    
    classes = {
        "Fake": "blue",
        "Real": "orange"
    }
    
    plot_histogram(data=data, label=label, classes=classes)
    plot_scatter(data=data, label=label, classes=classes)
    plot_correlation_matrix(data=data, label=label)
    plot_pca_explained_variance(data=data)
    plot_lda_histogram(data=data, label=label, classes=classes)

def plot_overall_DCF(min_DCFs, norm_DCFs, ls, title:str=""):
    for i, l in enumerate(ls):
        plt.plot(l, norm_DCFs[i], color = 'blue', marker = 'o')
        plt.plot(l, min_DCFs[i], color = 'orange', marker = 'o')
        
    plt.xscale('log', base=10)
    plt.xlabel('Regularization parameter (λ/C)')
    plt.ylabel('DCF')
    plt.title(f'Overall DCF for {title}')
    plt.legend(['Normalized DCF', 'MinDCF'])
    plt.show()

def plot_overall_DCF_rbf(min_DCFs, norm_DCFs, Cs, gammas):
    norm_cmap = cm.Blues  # Scale of blues for normalized DCF
    min_cmap = cm.Reds    # Scale of reds for minDCF

    # Get color scale based on the number of gammas
    colors_norm = norm_cmap(np.linspace(0.5, 1, len(gammas)))
    colors_min = min_cmap(np.linspace(0.5, 1, len(gammas))) 

    plt.figure(figsize=(10, 6))

    # Plot lines with colors
    for i, g in enumerate(gammas):
        plt.plot(Cs, norm_DCFs[g], marker='o', linestyle='-', color=colors_norm[i], label=f'Norm DCF (Gamma={g:.1e})')
        plt.plot(Cs, min_DCFs[g], marker='o', linestyle='-', color=colors_min[i], label=f'Min DCF (Gamma={g:.1e})')

    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title('Overall DCF by Gamma and C')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid()
    plt.show()
    
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