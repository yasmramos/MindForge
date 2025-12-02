package com.mindforge.clustering;

import java.io.Serializable;
import java.util.*;

/**
 * Agglomerative Hierarchical Clustering.
 * 
 * Builds a hierarchy of clusters using a bottom-up approach,
 * starting with each point as its own cluster and merging
 * the closest clusters iteratively.
 * 
 * @author MindForge
 */
public class HierarchicalClustering implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum Linkage {
        SINGLE,    // Minimum distance between clusters
        COMPLETE,  // Maximum distance between clusters
        AVERAGE,   // Average distance between clusters
        WARD       // Ward's minimum variance
    }
    
    private int nClusters;
    private Linkage linkage;
    private double distanceThreshold;
    
    private int[] labels;
    private double[][] linkageMatrix;
    private boolean fitted;
    
    /**
     * Creates a HierarchicalClustering with default parameters.
     */
    public HierarchicalClustering() {
        this(2, Linkage.WARD);
    }
    
    /**
     * Creates a HierarchicalClustering with specified number of clusters.
     */
    public HierarchicalClustering(int nClusters) {
        this(nClusters, Linkage.WARD);
    }
    
    /**
     * Creates a HierarchicalClustering with specified parameters.
     */
    public HierarchicalClustering(int nClusters, Linkage linkage) {
        if (nClusters < 1) {
            throw new IllegalArgumentException("nClusters must be at least 1");
        }
        
        this.nClusters = nClusters;
        this.linkage = linkage;
        this.distanceThreshold = -1;
        this.fitted = false;
    }
    
    /**
     * Creates with distance threshold instead of nClusters.
     */
    public HierarchicalClustering(double distanceThreshold, Linkage linkage) {
        if (distanceThreshold <= 0) {
            throw new IllegalArgumentException("distanceThreshold must be positive");
        }
        
        this.nClusters = -1;
        this.distanceThreshold = distanceThreshold;
        this.linkage = linkage;
        this.fitted = false;
    }
    
    /**
     * Performs hierarchical clustering.
     */
    public int[] fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("X cannot be null or empty");
        }
        
        int n = X.length;
        
        // Initialize clusters (each point is its own cluster)
        List<List<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> cluster = new ArrayList<>();
            cluster.add(i);
            clusters.add(cluster);
        }
        
        // Compute initial distance matrix
        double[][] distMatrix = computeDistanceMatrix(X);
        
        // Linkage matrix for dendrogram
        linkageMatrix = new double[n - 1][4]; // [cluster1, cluster2, distance, size]
        int mergeStep = 0;
        
        // Active cluster indices
        Set<Integer> activeIndices = new HashSet<>();
        for (int i = 0; i < n; i++) {
            activeIndices.add(i);
        }
        
        // Merge clusters until we reach desired number
        int targetClusters = nClusters > 0 ? nClusters : 1;
        
        while (activeIndices.size() > targetClusters) {
            // Find closest pair of clusters
            double minDist = Double.MAX_VALUE;
            int minI = -1, minJ = -1;
            
            List<Integer> active = new ArrayList<>(activeIndices);
            
            for (int i = 0; i < active.size(); i++) {
                for (int j = i + 1; j < active.size(); j++) {
                    int ci = active.get(i);
                    int cj = active.get(j);
                    
                    double dist = computeClusterDistance(clusters.get(ci), clusters.get(cj), X, distMatrix);
                    
                    if (dist < minDist) {
                        minDist = dist;
                        minI = ci;
                        minJ = cj;
                    }
                }
            }
            
            // Check distance threshold
            if (distanceThreshold > 0 && minDist > distanceThreshold) {
                break;
            }
            
            // Record merge in linkage matrix
            if (mergeStep < linkageMatrix.length) {
                linkageMatrix[mergeStep][0] = minI;
                linkageMatrix[mergeStep][1] = minJ;
                linkageMatrix[mergeStep][2] = minDist;
                linkageMatrix[mergeStep][3] = clusters.get(minI).size() + clusters.get(minJ).size();
            }
            mergeStep++;
            
            // Merge clusters
            clusters.get(minI).addAll(clusters.get(minJ));
            clusters.set(minJ, new ArrayList<>()); // Clear merged cluster
            activeIndices.remove(minJ);
        }
        
        // Assign labels
        labels = new int[n];
        int labelId = 0;
        for (int i : activeIndices) {
            for (int idx : clusters.get(i)) {
                labels[idx] = labelId;
            }
            labelId++;
        }
        
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
     * Computes distance between two clusters based on linkage method.
     */
    private double computeClusterDistance(List<Integer> c1, List<Integer> c2, 
                                          double[][] X, double[][] distMatrix) {
        switch (linkage) {
            case SINGLE:
                return singleLinkage(c1, c2, distMatrix);
            case COMPLETE:
                return completeLinkage(c1, c2, distMatrix);
            case AVERAGE:
                return averageLinkage(c1, c2, distMatrix);
            case WARD:
                return wardLinkage(c1, c2, X);
            default:
                return singleLinkage(c1, c2, distMatrix);
        }
    }
    
    private double singleLinkage(List<Integer> c1, List<Integer> c2, double[][] distMatrix) {
        double minDist = Double.MAX_VALUE;
        for (int i : c1) {
            for (int j : c2) {
                minDist = Math.min(minDist, distMatrix[i][j]);
            }
        }
        return minDist;
    }
    
    private double completeLinkage(List<Integer> c1, List<Integer> c2, double[][] distMatrix) {
        double maxDist = 0;
        for (int i : c1) {
            for (int j : c2) {
                maxDist = Math.max(maxDist, distMatrix[i][j]);
            }
        }
        return maxDist;
    }
    
    private double averageLinkage(List<Integer> c1, List<Integer> c2, double[][] distMatrix) {
        double sumDist = 0;
        for (int i : c1) {
            for (int j : c2) {
                sumDist += distMatrix[i][j];
            }
        }
        return sumDist / (c1.size() * c2.size());
    }
    
    private double wardLinkage(List<Integer> c1, List<Integer> c2, double[][] X) {
        int d = X[0].length;
        
        // Compute centroids
        double[] centroid1 = new double[d];
        double[] centroid2 = new double[d];
        
        for (int i : c1) {
            for (int j = 0; j < d; j++) {
                centroid1[j] += X[i][j];
            }
        }
        for (int j = 0; j < d; j++) {
            centroid1[j] /= c1.size();
        }
        
        for (int i : c2) {
            for (int j = 0; j < d; j++) {
                centroid2[j] += X[i][j];
            }
        }
        for (int j = 0; j < d; j++) {
            centroid2[j] /= c2.size();
        }
        
        // Ward's distance
        double dist = 0;
        for (int j = 0; j < d; j++) {
            dist += (centroid1[j] - centroid2[j]) * (centroid1[j] - centroid2[j]);
        }
        
        return Math.sqrt(dist) * Math.sqrt((2.0 * c1.size() * c2.size()) / (c1.size() + c2.size()));
    }
    
    /**
     * Computes pairwise distance matrix.
     */
    private double[][] computeDistanceMatrix(double[][] X) {
        int n = X.length;
        double[][] distances = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = 0;
                for (int k = 0; k < X[i].length; k++) {
                    dist += (X[i][k] - X[j][k]) * (X[i][k] - X[j][k]);
                }
                dist = Math.sqrt(dist);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        return distances;
    }
    
    // Getters
    public int[] getLabels() {
        return labels != null ? labels.clone() : null;
    }
    
    public double[][] getLinkageMatrix() {
        if (linkageMatrix == null) return null;
        double[][] copy = new double[linkageMatrix.length][];
        for (int i = 0; i < linkageMatrix.length; i++) {
            copy[i] = linkageMatrix[i].clone();
        }
        return copy;
    }
    
    public int getNClusters() {
        if (labels == null) return 0;
        Set<Integer> unique = new HashSet<>();
        for (int label : labels) {
            unique.add(label);
        }
        return unique.size();
    }
    
    public Linkage getLinkage() {
        return linkage;
    }
    
    public boolean isFitted() {
        return fitted;
    }
}
