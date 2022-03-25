# de_genes.py
# HW2, Computational Genomics, Spring 2022
# andrewid:

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np


# Do not change this function signature

def bh(genes, pvals, alpha):
    """(list, list, float) -> numpy array
    applies benjamini-hochberg procedure
    
    Parameters
    ----------
    genes: name of genes 
    pvalues: corresponding pvals
    alpha: desired false discovery rate
    
    Returns
    -------
    array containing gene names of significant genes.
    gene names do not need to be in any specific order.
    """

    numGenes = len(genes)
    sig_genes = []
    indices = np.argsort(pvals)
    
    for i, gene in enumerate(indices):
        
        if pvals[gene] <= (i+1)/numGenes * alpha:
            sig_genes.append(genes[indices[gene]])

    return np.asarray(sig_genes, dtype='str')

# define any helper function here    

def gene_dispersion(data, labels):

    ## Index case vs. control samples
    case_idxs = np.where(labels == 1)
    control_idxs = np.where(labels == 2)

    ## No. of genes, case samples, and control samples
    numGenes = len(data)
    numCases, numControls = len(case_idxs), len(control_idxs)

    ## Initiate independent gene exp datasets for cases & controls
    case_samples = np.ndarray(shape=(numGenes, numCases))
    control_samples = np.ndarray(shape=(numGenes, numCases))

    ## Add gene exp data for case & control samples
    for i in range(numCases):
        case_samples[:, i] = data[:, case_idxs[i]]
    for i in range(numControls):
        control_samples[:, i] = data[:, control_idxs[i]] 

    ## Calc. dispersion of each gene in the case & control datasets
    disp_case, disp_control = np.zeros(shape=(1, numGenes)), np.zeros(shape=(1, numGenes))
    for i in range(numGenes):
        disp_case[i] = np.std(case_samples[i, :])/np.mean(case_samples[i, :])
        disp_control[i] = np.std(control_samples[i, :])/np.mean(control_samples[i, :])

    return disp_case, disp_control

def compute_log_fold(gene_names, data1, data2):
    
    numGenes = len(gene_names)
    gene_fold_changes = []

    for i in range(numGenes):
        gene_fold_changes[i] = np.log2(data1[i] / data2[i])

    top_pos_genes = gene_names[np.argsort(gene_fold_changes)[::-1][:10]]
    top_neg_genes = gene_names[np.argsort(gene_fold_changes)[:10]]

    return top_pos_genes, top_neg_genes

def compute_mean_counts(data):
    mean_counts = np.mean(data, axis=1)
    return mean_counts
    
if __name__=="__main__":
    # Here is a free test case
    genes=['a', 'b', 'c']
    input1 = [0.01, 0.04, 0.1]
    print(bh(genes, input1, 0.05))
