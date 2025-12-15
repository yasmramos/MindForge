package io.github.yasmramos.mindforge.clustering;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Affinity Propagation clustering algorithm.
 * Creates clusters by sending messages between pairs of samples.
 */
public class AffinityPropagation implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final double damping;
    private final int maxIterations;
    private final int convergenceIter;
    private double preference;
    private int[] labels;
    private int[] clusterCentersIndices;
    
    public AffinityPropagation(double damping, int maxIterations, int convergenceIter, double preference) {
        this.damping = damping;
        this.maxIterations = maxIterations;
        this.convergenceIter = convergenceIter;
        this.preference = preference;
    }
    
    public AffinityPropagation() {
        this(0.5, 200, 15, Double.MIN_VALUE);
    }
    
    public int[] fitPredict(double[][] X) {
        fit(X);
        return labels;
    }
    
    public void fit(double[][] X) {
        int n = X.length;
        
        // Compute similarity matrix
        double[][] S = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                S[i][j] = -squaredDistance(X[i], X[j]);
            }
        }
        
        // Set preference (diagonal) - median of similarities if not set
        if (preference == Double.MIN_VALUE) {
            double[] similarities = new double[n * (n - 1) / 2];
            int idx = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    similarities[idx++] = S[i][j];
                }
            }
            Arrays.sort(similarities);
            preference = similarities[similarities.length / 2];
        }
        for (int i = 0; i < n; i++) {
            S[i][i] = preference;
        }
        
        // Initialize responsibility and availability matrices
        double[][] R = new double[n][n];
        double[][] A = new double[n][n];
        
        int[] lastLabels = new int[n];
        int stableCount = 0;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // Update responsibilities
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int kPrime = 0; kPrime < n; kPrime++) {
                        if (kPrime != k) {
                            max = Math.max(max, A[i][kPrime] + S[i][kPrime]);
                        }
                    }
                    double newR = S[i][k] - max;
                    R[i][k] = damping * R[i][k] + (1 - damping) * newR;
                }
            }
            
            // Update availabilities
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    if (i == k) {
                        double sum = 0;
                        for (int iPrime = 0; iPrime < n; iPrime++) {
                            if (iPrime != k) {
                                sum += Math.max(0, R[iPrime][k]);
                            }
                        }
                        double newA = sum;
                        A[i][k] = damping * A[i][k] + (1 - damping) * newA;
                    } else {
                        double sum = 0;
                        for (int iPrime = 0; iPrime < n; iPrime++) {
                            if (iPrime != i && iPrime != k) {
                                sum += Math.max(0, R[iPrime][k]);
                            }
                        }
                        double newA = Math.min(0, R[k][k] + sum);
                        A[i][k] = damping * A[i][k] + (1 - damping) * newA;
                    }
                }
            }
            
            // Check convergence
            int[] currentLabels = extractLabels(R, A, n);
            if (Arrays.equals(currentLabels, lastLabels)) {
                stableCount++;
                if (stableCount >= convergenceIter) break;
            } else {
                stableCount = 0;
                lastLabels = currentLabels;
            }
        }
        
        labels = extractLabels(R, A, n);
        extractClusterCenters(n);
    }
    
    private int[] extractLabels(double[][] R, double[][] A, int n) {
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            double maxVal = Double.NEGATIVE_INFINITY;
            int maxK = 0;
            for (int k = 0; k < n; k++) {
                double val = R[i][k] + A[i][k];
                if (val > maxVal) {
                    maxVal = val;
                    maxK = k;
                }
            }
            labels[i] = maxK;
        }
        return labels;
    }
    
    private void extractClusterCenters(int n) {
        boolean[] isCenter = new boolean[n];
        for (int label : labels) {
            isCenter[label] = true;
        }
        int count = 0;
        for (boolean b : isCenter) if (b) count++;
        clusterCentersIndices = new int[count];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (isCenter[i]) {
                clusterCentersIndices[idx++] = i;
            }
        }
        
        // Relabel to consecutive integers
        int[] mapping = new int[n];
        Arrays.fill(mapping, -1);
        int labelIdx = 0;
        for (int center : clusterCentersIndices) {
            mapping[center] = labelIdx++;
        }
        for (int i = 0; i < labels.length; i++) {
            labels[i] = mapping[labels[i]];
        }
    }
    
    private double squaredDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    public int[] getLabels() { return labels; }
    public int[] getClusterCentersIndices() { return clusterCentersIndices; }
    
    public static class Builder {
        private double damping = 0.5;
        private int maxIterations = 200;
        private int convergenceIter = 15;
        private double preference = Double.MIN_VALUE;
        
        public Builder damping(double d) { this.damping = d; return this; }
        public Builder maxIterations(int n) { this.maxIterations = n; return this; }
        public Builder convergenceIter(int n) { this.convergenceIter = n; return this; }
        public Builder preference(double p) { this.preference = p; return this; }
        public AffinityPropagation build() { return new AffinityPropagation(damping, maxIterations, convergenceIter, preference); }
    }
}
