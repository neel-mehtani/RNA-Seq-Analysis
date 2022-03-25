# classification.py
# HW2, Computational Genomics, Spring 2022
# andrewid: nmehtani

# WARNING: Do not change the file name; Autograder expects it.

import sys

import numpy as np
from scipy.sparse import csc_matrix, save_npz, load_npz

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def get_top_gene_filter(data, n_keep = 2000):
    """Select top n_keep most dispersed genes.

    Args:
        data (n x m matrix): input gene expression data of shape num_cells x num_genes
        n_keep (int): number of genes to be kepted after filtration; default 2000

    Returns:
        filter (array of length n_keep): an array of column indices that can be used as an
            index to keep only certain genes in data. Each element of filter is the column
            index of a highly-dispersed gene in data.
    """

    numGenes = data.shape[1]
    dispersions = np.zeros(numGenes)
    for i in range(numGenes):
        currGene = data[:, i]
        disp_i = compute_gene_dispersion(currGene)
        dispersions.append(disp_i)
    top_gene_idxs = np.argsort()[::-1][:n_keep]

    return top_gene_idxs


def reduce_dimensionality_pca(filtered_train_gene_expression, filtered_test_gene_expression, n_components = 20):
    """Train a PCA model and use it to reduce the training and testing data.
    
    Args:
        filtered_train_gene_expression (n_train x num_top_genes matrix): input filtered training expression data 
        filtered_test_gene_expression (n_test x num_top_genes matrix): input filtered test expression data 
        
    Return:
        (reduced_train_data, reduced_test_data): a tuple of
            1. The filtered training data transformed to the PC space.
            2. The filtered test data transformed to the PC space.
    """
    
    pipeline = make_pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])
    reduced_train_data = pipeline.fit_transform(filtered_train_gene_expression)
    reduced_test_data = pipeline.fit_transform(filtered_test_gene_expression)

    return (reduced_train_data, reduced_test_data)


def plot_transformed_cells(reduced_train_data, train_labels):
    """Plot the PCA-reduced training data using just the first 2 principal components.
    
    Args:
        reduced_train_data (n_train x num_components matrix): reduced training expression data
        train_labels (array of length n_train): array of cell type labels for training data
        
    Return:
        None

    """
    ## Keep top-N components explaining for x% of variance
    x1, x2 = reduced_train_data[:, 0], reduced_train_data[:, 1]

    ## 2D PCA Analysis
    targets = list(set(list(train_labels)))
    colors = np.random.rand(len(targets),3)

    plt.figure()
    plt.figure(figsize=(10,8))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("2D Principal Component Analysis of Dataset",fontsize=20)
    
    for target, color in zip(targets,colors):
        indicesToKeep = np.where(train_labels == target)
        plt.scatter(x1[indicesToKeep], x2[indicesToKeep], c = color, s = 50)

    plt.legend(targets,prop={'size': 15})
    plt.show()
    plt.close()
    
    return 

def train_and_evaluate_svm_classifier(reduced_train_data, reduced_test_data, train_labels, test_labels):
    """Train and evaluate a simple SVM-based classification pipeline.
    
    Before passing the data to the SVM module, this function scales the data such that the mean
    is 0 and the variance is 1.
    
    Args:
        reduced_train_data (n_train x num_components matrix): reduced training expression data
        train_labels (array of length n_train): array of cell type labels for training data
        
    Return:
        (classifier, score): a tuple consisting of
            1. classifier: the trained classifier
            2. The score (accuracy) of the classifier on the test data.

    """

    ## encode test labels to numbers?

    clf = svm.SVC(kernel='poly', degree=3, C=1).fit(reduced_train_data, train_labels)
    clf_preds = clf.predict(reduced_test_data)

    accuracy = accuracy_score(clf_preds, test_labels)
    
    return (clf, accuracy)
  
if __name__ == "__main__":
    
    train_gene_expression = load_npz(sys.argv[1]).toarray()
    test_gene_expression = load_npz(sys.argv[2]).toarray()
    train_labels = np.load(sys.argv[3])
    test_labels = np.load(sys.argv[4])
    
    top_gene_filter = get_top_gene_filter(train_gene_expression)
    filtered_train_gene_expression = train_gene_expression[:, top_gene_filter]
    filtered_test_gene_expression = test_gene_expression[:, top_gene_filter]
        
    mode = sys.argv[5]
    if mode == "svm_pipeline":
        # TODO: Implement the pipeline here

        reduced_train_data, reduced_test_data = reduce_dimensionality_pca(filtered_train_gene_expression, filtered_test_gene_expression)
        
        model, test_acc = train_and_evaluate_svm_classifier(reduced_train_data, reduced_test_data, train_labels, test_labels)
        
        print("Training accuracy: %.3f"%model.score(reduced_train_data, train_labels))
        print("Test accuracy: %.3f"%test_acc)
