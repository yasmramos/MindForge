package com.mindforge.clustering;

/**
 * Interface for clustering algorithms.
 * A clusterer groups similar data points together.
 * 
 * @param <T> the type of input data
 */
public interface Clusterer<T> {
    
    /**
     * Performs clustering on the given data.
     * 
     * @param x data to cluster
     * @return cluster assignments for each data point
     */
    int[] cluster(T[] x);
    
    /**
     * Returns the number of clusters.
     * 
     * @return number of clusters
     */
    int getNumClusters();
    
    /**
     * Predicts the cluster assignment for a new data point.
     * 
     * @param x data point
     * @return cluster assignment
     */
    int predict(T x);
}
