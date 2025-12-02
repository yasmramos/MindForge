package com.mindforge.clustering;

import java.io.Serializable;
import java.util.*;

/**
 * Mean Shift Clustering.
 * 
 * A non-parametric clustering algorithm that discovers
 * cluster centers by shifting points towards high-density regions.
 * 
 * @author MindForge
 */
public class MeanShift implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private double bandwidth;
    private int maxIterations;
    private double tolerance;
    private boolean binSeeding;
    
    private int[] labels;
    private double[][] clusterCenters;
    private int nClusters;
    private boolean fitted;
    
    /**
     * Creates a MeanShift with default parameters.
     */
    public MeanShift() {
        this(-1, 300, 1e-4, false);
    }
    
    /**
     * Creates a MeanShift with specified bandwidth.
     */
    public MeanShift(double bandwidth) {
        this(bandwidth, 300, 1e-4, false);
    }
    
    /**
     * Creates a MeanShift with full configuration.
     * 
     * @param bandwidth Kernel bandwidth (-1 for auto-estimation)
     * @param maxIterations Maximum iterations for convergence
     * @param tolerance Convergence threshold
     * @param binSeeding Whether to use binning for initial seeding
     */
    public MeanShift(double bandwidth, int maxIterations, double tolerance, boolean binSeeding) {
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive");
        }
        if (tolerance <= 0) {
            throw new IllegalArgumentException("tolerance must be positive");
        }
        
        this.bandwidth = bandwidth;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.binSeeding = binSeeding;
        this.fitted = false;
    }
    
    /**
     * Performs mean shift clustering.
     */
    public int[] fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("X cannot be null or empty");
        }
        
        int n = X.length;
        int d = X[0].length;
        
        // Estimate bandwidth if not provided
        double actualBandwidth = bandwidth > 0 ? bandwidth : estimateBandwidth(X);
        
        // Initialize seeds
        List<double[]> seeds = new ArrayList<>();
        if (binSeeding) {
            seeds = getBinnedSeeds(X, actualBandwidth);
        } else {
            for (double[] point : X) {
                seeds.add(point.clone());
            }
        }
        
        // Perform mean shift for each seed
        List<double[]> convergedCenters = new ArrayList<>();
        
        for (double[] seed : seeds) {
            double[] center = seed.clone();
            
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] newCenter = new double[d];
                double weightSum = 0;
                
                // Calculate weighted mean
                for (double[] point : X) {
                    double dist = euclideanDistance(center, point);
                    double weight = gaussianKernel(dist, actualBandwidth);
                    
                    for (int j = 0; j < d; j++) {
                        newCenter[j] += weight * point[j];
                    }
                    weightSum += weight;
                }
                
                if (weightSum > 0) {
                    for (int j = 0; j < d; j++) {
                        newCenter[j] /= weightSum;
                    }
                }
                
                // Check convergence
                double shift = euclideanDistance(center, newCenter);
                center = newCenter;
                
                if (shift < tolerance) {
                    break;
                }
            }
            
            // Add to converged centers if not duplicate
            boolean isDuplicate = false;
            for (double[] existing : convergedCenters) {
                if (euclideanDistance(existing, center) < actualBandwidth / 2) {
                    isDuplicate = true;
                    break;
                }
            }
            
            if (!isDuplicate) {
                convergedCenters.add(center);
            }
        }
        
        // Convert to array
        nClusters = convergedCenters.size();
        clusterCenters = new double[nClusters][d];
        for (int i = 0; i < nClusters; i++) {
            clusterCenters[i] = convergedCenters.get(i);
        }
        
        // Assign labels based on nearest center
        labels = new int[n];
        for (int i = 0; i < n; i++) {
            double minDist = Double.MAX_VALUE;
            int nearestCluster = 0;
            
            for (int c = 0; c < nClusters; c++) {
                double dist = euclideanDistance(X[i], clusterCenters[c]);
                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = c;
                }
            }
            
            labels[i] = nearestCluster;
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
     * Predicts cluster labels for new data.
     */
    public int[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model not fitted");
        }
        
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double minDist = Double.MAX_VALUE;
            int nearestCluster = 0;
            
            for (int c = 0; c < nClusters; c++) {
                double dist = euclideanDistance(X[i], clusterCenters[c]);
                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = c;
                }
            }
            
            predictions[i] = nearestCluster;
        }
        
        return predictions;
    }
    
    /**
     * Estimates bandwidth using mean of pairwise distances.
     */
    private double estimateBandwidth(double[][] X) {
        int n = Math.min(X.length, 100); // Sample for efficiency
        double sum = 0;
        int count = 0;
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                sum += euclideanDistance(X[i], X[j]);
                count++;
            }
        }
        
        return count > 0 ? sum / count * 0.3 : 1.0;
    }
    
    /**
     * Gets binned seeds for faster convergence.
     */
    private List<double[]> getBinnedSeeds(double[][] X, double bandwidth) {
        int d = X[0].length;
        
        // Find bounds
        double[] mins = new double[d];
        double[] maxs = new double[d];
        Arrays.fill(mins, Double.MAX_VALUE);
        Arrays.fill(maxs, Double.MIN_VALUE);
        
        for (double[] point : X) {
            for (int j = 0; j < d; j++) {
                mins[j] = Math.min(mins[j], point[j]);
                maxs[j] = Math.max(maxs[j], point[j]);
            }
        }
        
        // Create bins
        Map<String, double[]> bins = new HashMap<>();
        Map<String, Integer> binCounts = new HashMap<>();
        
        for (double[] point : X) {
            StringBuilder keyBuilder = new StringBuilder();
            for (int j = 0; j < d; j++) {
                int bin = (int) ((point[j] - mins[j]) / bandwidth);
                keyBuilder.append(bin).append(",");
            }
            String key = keyBuilder.toString();
            
            if (!bins.containsKey(key)) {
                bins.put(key, new double[d]);
                binCounts.put(key, 0);
            }
            
            double[] binSum = bins.get(key);
            for (int j = 0; j < d; j++) {
                binSum[j] += point[j];
            }
            binCounts.put(key, binCounts.get(key) + 1);
        }
        
        // Compute bin centers
        List<double[]> seeds = new ArrayList<>();
        for (String key : bins.keySet()) {
            double[] binSum = bins.get(key);
            int count = binCounts.get(key);
            double[] center = new double[d];
            for (int j = 0; j < d; j++) {
                center[j] = binSum[j] / count;
            }
            seeds.add(center);
        }
        
        return seeds;
    }
    
    /**
     * Gaussian kernel function.
     */
    private double gaussianKernel(double distance, double bandwidth) {
        return Math.exp(-0.5 * (distance / bandwidth) * (distance / bandwidth));
    }
    
    /**
     * Euclidean distance between two points.
     */
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return Math.sqrt(sum);
    }
    
    // Getters
    public int[] getLabels() {
        return labels != null ? labels.clone() : null;
    }
    
    public double[][] getClusterCenters() {
        if (clusterCenters == null) return null;
        double[][] copy = new double[clusterCenters.length][];
        for (int i = 0; i < clusterCenters.length; i++) {
            copy[i] = clusterCenters[i].clone();
        }
        return copy;
    }
    
    public int getNClusters() {
        return nClusters;
    }
    
    public double getBandwidth() {
        return bandwidth;
    }
    
    public boolean isFitted() {
        return fitted;
    }
}
