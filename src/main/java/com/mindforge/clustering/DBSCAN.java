package com.mindforge.clustering;

import java.io.Serializable;
import java.util.*;

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
 * 
 * Clusters points based on density, can find arbitrarily shaped clusters
 * and identifies outliers as noise points.
 * 
 * @author MindForge
 */
public class DBSCAN implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private double eps;
    private int minSamples;
    private String metric;
    
    private int[] labels;
    private int nClusters;
    private int[] coreIndices;
    private boolean fitted;
    
    private static final int NOISE = -1;
    private static final int UNDEFINED = -2;
    
    /**
     * Creates a DBSCAN with default parameters.
     */
    public DBSCAN() {
        this(0.5, 5, "euclidean");
    }
    
    /**
     * Creates a DBSCAN with specified eps and minSamples.
     * 
     * @param eps Maximum distance between points in a neighborhood
     * @param minSamples Minimum points required to form a dense region
     */
    public DBSCAN(double eps, int minSamples) {
        this(eps, minSamples, "euclidean");
    }
    
    /**
     * Creates a DBSCAN with full configuration.
     */
    public DBSCAN(double eps, int minSamples, String metric) {
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive");
        }
        if (minSamples < 1) {
            throw new IllegalArgumentException("minSamples must be at least 1");
        }
        
        this.eps = eps;
        this.minSamples = minSamples;
        this.metric = metric;
        this.fitted = false;
    }
    
    /**
     * Performs DBSCAN clustering.
     * 
     * @param X Data matrix
     * @return Cluster labels (-1 for noise)
     */
    public int[] fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("X cannot be null or empty");
        }
        
        int n = X.length;
        labels = new int[n];
        Arrays.fill(labels, UNDEFINED);
        
        // Precompute distances
        double[][] distances = computeDistanceMatrix(X);
        
        // Find neighbors for each point
        List<List<Integer>> neighborhoods = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> neighbors = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (distances[i][j] <= eps) {
                    neighbors.add(j);
                }
            }
            neighborhoods.add(neighbors);
        }
        
        // Find core points
        List<Integer> corePointsList = new ArrayList<>();
        boolean[] isCore = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (neighborhoods.get(i).size() >= minSamples) {
                isCore[i] = true;
                corePointsList.add(i);
            }
        }
        coreIndices = corePointsList.stream().mapToInt(Integer::intValue).toArray();
        
        // Cluster assignment
        int clusterId = 0;
        
        for (int i = 0; i < n; i++) {
            if (labels[i] != UNDEFINED) continue;
            
            List<Integer> neighbors = neighborhoods.get(i);
            
            if (neighbors.size() < minSamples) {
                labels[i] = NOISE;
                continue;
            }
            
            // Expand cluster
            labels[i] = clusterId;
            Queue<Integer> seedSet = new LinkedList<>(neighbors);
            seedSet.remove(Integer.valueOf(i));
            
            while (!seedSet.isEmpty()) {
                int q = seedSet.poll();
                
                if (labels[q] == NOISE) {
                    labels[q] = clusterId;
                }
                
                if (labels[q] != UNDEFINED) continue;
                
                labels[q] = clusterId;
                List<Integer> qNeighbors = neighborhoods.get(q);
                
                if (qNeighbors.size() >= minSamples) {
                    for (int neighbor : qNeighbors) {
                        if (labels[neighbor] == UNDEFINED || labels[neighbor] == NOISE) {
                            if (!seedSet.contains(neighbor)) {
                                seedSet.add(neighbor);
                            }
                        }
                    }
                }
            }
            
            clusterId++;
        }
        
        nClusters = clusterId;
        fitted = true;
        
        return labels.clone();
    }
    
    /**
     * Fits and returns labels.
     */
    public int[] fitPredict(double[][] X) {
        return fit(X);
    }
    
    /**
     * Computes distance matrix.
     */
    private double[][] computeDistanceMatrix(double[][] X) {
        int n = X.length;
        double[][] distances = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = computeDistance(X[i], X[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        return distances;
    }
    
    /**
     * Computes distance between two points.
     */
    private double computeDistance(double[] a, double[] b) {
        if (metric.equals("manhattan")) {
            double dist = 0;
            for (int i = 0; i < a.length; i++) {
                dist += Math.abs(a[i] - b[i]);
            }
            return dist;
        } else { // euclidean
            double dist = 0;
            for (int i = 0; i < a.length; i++) {
                dist += (a[i] - b[i]) * (a[i] - b[i]);
            }
            return Math.sqrt(dist);
        }
    }
    
    // Getters
    public int[] getLabels() {
        return labels != null ? labels.clone() : null;
    }
    
    public int getNClusters() {
        return nClusters;
    }
    
    public int[] getCoreIndices() {
        return coreIndices != null ? coreIndices.clone() : null;
    }
    
    public int getNumNoise() {
        if (labels == null) return 0;
        int count = 0;
        for (int label : labels) {
            if (label == NOISE) count++;
        }
        return count;
    }
    
    public double getEps() {
        return eps;
    }
    
    public int getMinSamples() {
        return minSamples;
    }
    
    public boolean isFitted() {
        return fitted;
    }
}
