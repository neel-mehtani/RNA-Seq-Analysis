# normalization.py
# HW2, Computational Genomics, Spring 2022
# andrewid: nmehtani

# WARNING: Do not change the file name; Autograder expects it.

import sys
import numpy as np
import matplotlib.pyplot as plt

PER_MILLION = 1/1000000
PER_KILOBASE = 1/1000

# Do not change this function signature
def rpkm(raw_counts, gene_lengths):
    
    """Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """
    
    norm_counts = raw_counts.copy()
    totalReads = np.sum(norm_counts, axis=0)
    
    numGenes, numSamples = raw_counts.shape[0], raw_counts.shape[1]
    
    for m in range(numSamples):
        
        M = totalReads[m]
        
        for n in range(numGenes):
            norm_counts[n, m] = (norm_counts[n, m] * 10**9)/(M * gene_lengths[n])

    return norm_counts
    
# Do not change this function signature
def tpm(raw_counts, gene_lengths):
    """
    Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """    
    numGenes, numSamples = raw_counts.shape[0], raw_counts.shape[1]
    norm_counts = raw_counts.copy()
    
    for n in range(numGenes):
        norm_counts[n, :] = norm_counts[n, :] / (gene_lengths[n]/10**3)
        
    scaling_factors = np.sum(norm_counts, axis = 0)
    scaling_factors = scaling_factors/1000000
      
    for m in range(numSamples):
        norm_counts[:, m] = norm_counts[:, m] / scaling_factors[m]
      
    return norm_counts
    
   
# define any helper function here    


# Do not change this function signature

def size_factor(raw_counts):

    """
    Find the normalized counts for raw_counts
    Returns: a matrix of same size as raw_counts
    """

    norm_counts = raw_counts.copy()
    
    numGenes, numSamples = raw_counts.shape[0], raw_counts.shape[1]
    
    sizes = []
    for m in range(numSamples):

        ratios = []
        for n in range(numGenes):
            denom = np.prod(raw_counts[n, :]) ** (1/numSamples)
            ratios.append(raw_counts[n][m]/denom)

        sizes.append(np.median(ratios))
        
    for m in range(numSamples):
        norm_counts[:, m] = norm_counts[:, m] / sizes[m]
   
    return norm_counts
            
if __name__=="__main__":
    raw_counts=np.loadtxt(sys.argv[1])
    gene_lengths=np.loadtxt(sys.argv[2])
    
    rpkm1=rpkm(raw_counts, gene_lengths)
    tpm1=tpm(raw_counts, gene_lengths)
    size_factor1=size_factor(raw_counts)

    # TODO: write plotting code here
    log_raw_counts = np.log2(raw_counts)
    log_rpkm = np.log2(rpkm1)
    log_tpm = np.log2(tpm1)
    log_size_factor = np.log2(size_factor1)
    np.savetxt("size_factor_normalized_counts.txt", log_size_factor)
    
    plt.figure(figsize=(10,6))
    plt.boxplot(x=log_raw_counts)
    plt.title("Log2 Raw Counts Boxplot")
    plt.xlabel("Sample ID")
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.boxplot(x=log_rpkm)
    plt.title("Log2 RPKM Boxplot")
    plt.xlabel("Sample ID")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.boxplot(x=log_tpm)
    plt.title("Log2 TPM Boxplot")
    plt.xlabel("Sample ID")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.boxplot(x=log_size_factor)
    plt.title("Log2 Size-Factor Boxplot")
    plt.xlabel("Sample ID")
    plt.show()
