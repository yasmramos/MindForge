package io.github.yasmramos.mindforge.online;

import io.github.yasmramos.mindforge.data.Dataset;

import java.util.Arrays;

/**
 * Online K-Means Clustering for streaming data.
 * Updates cluster centroids incrementally as new data arrives.
 */
public class OnlineKMeans {
    
    private int nClusters;
    private int nFeatures;
    private double[][] centroids;
    private int[] clusterCounts;
    private boolean initialized;
    
    public OnlineKMeans(int nClusters, int nFeatures) {
        this.nClusters = nClusters;
        this.nFeatures = nFeatures;
        this.centroids = new double[nClusters][nFeatures];
        this.clusterCounts = new int[nClusters];
        this.initialized = false;
    }
    
    public void partialFit(double[][] X) {
        if (!initialized) {
            initialize(X);
            return;
        }
        
        for (int i = 0; i < X.length; i++) {
            int cluster = predict(X[i]);
            clusterCounts[cluster]++;
            
            double alpha = 1.0 / clusterCounts[cluster];
            for (int j = 0; j < nFeatures; j++) {
                centroids[cluster][j] += alpha * (X[i][j] - centroids[cluster][j]);
            }
        }
    }
    
    private void initialize(double[][] X) {
        int n = Math.min(X.length, nClusters);
        for (int i = 0; i < n; i++) {
            System.arraycopy(X[i], 0, centroids[i], 0, nFeatures);
            clusterCounts[i] = 1;
        }
        
        for (int i = n; i < nClusters; i++) {
            int randomIdx = (int)(Math.random() * X.length);
            System.arraycopy(X[randomIdx], 0, centroids[i], 0, nFeatures);
            clusterCounts[i] = 0;
        }
        
        initialized = true;
    }
    
    public int predict(double[] x) {
        int closestCluster = 0;
        double minDistance = Double.MAX_VALUE;
        
        for (int c = 0; c < nClusters; c++) {
            if (clusterCounts[c] == 0) continue;
            
            double distance = 0.0;
            for (int j = 0; j < nFeatures; j++) {
                double diff = x[j] - centroids[c][j];
                distance += diff * diff;
            }
            
            if (distance < minDistance) {
                minDistance = distance;
                closestCluster = c;
            }
        }
        
        return closestCluster;
    }
    
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    public double[][] getCentroids() {
        double[][] result = new double[nClusters][];
        for (int c = 0; c < nClusters; c++) {
            result[c] = Arrays.copyOf(centroids[c], nFeatures);
        }
        return result;
    }
}
