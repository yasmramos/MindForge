package io.github.yasmramos.mindforge.clustering;

import io.github.yasmramos.mindforge.math.Distance;
import java.util.Arrays;
import java.util.Random;

/**
 * K-Means clustering algorithm.
 * Partitions data into k clusters by minimizing the sum of squared distances
 * from data points to their assigned cluster centers.
 */
public class KMeans implements Clusterer<double[]> {
    
    private final int k;
    private final int maxIterations;
    private final Random random;
    private double[][] centroids;
    private boolean fitted;
    
    /**
     * Creates a new K-Means clusterer with default parameters.
     * 
     * @param k number of clusters
     */
    public KMeans(int k) {
        this(k, 100, new Random());
    }
    
    /**
     * Creates a new K-Means clusterer.
     * 
     * @param k number of clusters
     * @param maxIterations maximum number of iterations
     * @param random random number generator for initialization
     */
    public KMeans(int k, int maxIterations, Random random) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive");
        }
        
        this.k = k;
        this.maxIterations = maxIterations;
        this.random = random;
        this.fitted = false;
    }
    
    @Override
    public int[] cluster(double[][] x) {
        if (x.length < k) {
            throw new IllegalArgumentException("Data size must be at least k");
        }
        
        int n = x.length;
        int dimensions = x[0].length;
        
        // Initialize centroids randomly
        centroids = new double[k][dimensions];
        int[] selectedIndices = new int[k];
        for (int i = 0; i < k; i++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (contains(selectedIndices, idx, i));
            selectedIndices[i] = idx;
            centroids[i] = Arrays.copyOf(x[idx], dimensions);
        }
        
        int[] assignments = new int[n];
        boolean changed = true;
        int iteration = 0;
        
        while (changed && iteration < maxIterations) {
            changed = false;
            
            // Assign each point to nearest centroid
            for (int i = 0; i < n; i++) {
                int nearestCluster = findNearestCentroid(x[i]);
                if (assignments[i] != nearestCluster) {
                    assignments[i] = nearestCluster;
                    changed = true;
                }
            }
            
            // Update centroids
            updateCentroids(x, assignments);
            iteration++;
        }
        
        fitted = true;
        return assignments;
    }
    
    @Override
    public int predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        return findNearestCentroid(x);
    }
    
    @Override
    public int getNumClusters() {
        return k;
    }
    
    /**
     * Returns the cluster centroids.
     * 
     * @return array of centroids
     */
    public double[][] getCentroids() {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        double[][] result = new double[centroids.length][];
        for (int i = 0; i < centroids.length; i++) {
            result[i] = Arrays.copyOf(centroids[i], centroids[i].length);
        }
        return result;
    }
    
    /**
     * Finds the nearest centroid for a given point.
     * 
     * @param x data point
     * @return index of nearest centroid
     */
    private int findNearestCentroid(double[] x) {
        double minDistance = Double.MAX_VALUE;
        int nearestCluster = 0;
        
        for (int i = 0; i < k; i++) {
            double distance = Distance.euclidean(x, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }
        
        return nearestCluster;
    }
    
    /**
     * Updates centroids based on current assignments.
     * 
     * @param x data points
     * @param assignments cluster assignments
     */
    private void updateCentroids(double[][] x, int[] assignments) {
        int dimensions = x[0].length;
        double[][] newCentroids = new double[k][dimensions];
        int[] counts = new int[k];
        
        // Sum up points in each cluster
        for (int i = 0; i < x.length; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int j = 0; j < dimensions; j++) {
                newCentroids[cluster][j] += x[i][j];
            }
        }
        
        // Calculate averages
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < dimensions; j++) {
                    newCentroids[i][j] /= counts[i];
                }
                centroids[i] = newCentroids[i];
            }
            // If a cluster is empty, keep the old centroid
        }
    }
    
    /**
     * Checks if an array contains a value up to a certain index.
     * 
     * @param arr array to check
     * @param value value to find
     * @param upTo check up to this index (exclusive)
     * @return true if value is found
     */
    private boolean contains(int[] arr, int value, int upTo) {
        for (int i = 0; i < upTo; i++) {
            if (arr[i] == value) {
                return true;
            }
        }
        return false;
    }
}
