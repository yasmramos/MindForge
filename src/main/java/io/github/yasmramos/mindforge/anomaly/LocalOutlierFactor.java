package io.github.yasmramos.mindforge.anomaly;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Local Outlier Factor (LOF) algorithm for anomaly detection.
 * Measures local deviation of density with respect to neighbors.
 */
public class LocalOutlierFactor implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nNeighbors;
    private final double contamination;
    private double[][] trainData;
    private double threshold;
    
    public LocalOutlierFactor(int nNeighbors, double contamination) {
        this.nNeighbors = nNeighbors;
        this.contamination = contamination;
    }
    
    public LocalOutlierFactor() {
        this(20, 0.1);
    }
    
    public void fit(double[][] X) {
        this.trainData = X;
        double[] scores = computeLOFScores(X);
        double[] sortedScores = scores.clone();
        Arrays.sort(sortedScores);
        int thresholdIdx = (int) ((1 - contamination) * X.length);
        this.threshold = sortedScores[Math.min(thresholdIdx, X.length - 1)];
    }
    
    public int[] predict(double[][] X) {
        double[] scores = computeLOFScores(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = scores[i] > threshold ? -1 : 1;
        }
        return predictions;
    }
    
    public int[] fitPredict(double[][] X) {
        fit(X);
        return predict(X);
    }
    
    public double[] decisionFunction(double[][] X) {
        double[] scores = computeLOFScores(X);
        for (int i = 0; i < scores.length; i++) {
            scores[i] = -scores[i];
        }
        return scores;
    }
    
    private double[] computeLOFScores(double[][] X) {
        int n = X.length;
        double[][] distances = new double[n][n];
        int[][] neighbors = new int[n][nNeighbors];
        double[] kDistances = new double[n];
        
        // Compute distances
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double dist = euclideanDistance(X[i], X[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        
        // Find k-nearest neighbors
        for (int i = 0; i < n; i++) {
            int[] indices = new int[n];
            double[] dists = new double[n];
            for (int j = 0; j < n; j++) {
                indices[j] = j;
                dists[j] = distances[i][j];
            }
            sortByDistance(indices, dists);
            for (int k = 0; k < nNeighbors; k++) {
                neighbors[i][k] = indices[k + 1]; // Skip self
            }
            kDistances[i] = dists[nNeighbors];
        }
        
        // Compute reachability distances and LRD
        double[] lrd = new double[n];
        for (int i = 0; i < n; i++) {
            double sumReachDist = 0;
            for (int j = 0; j < nNeighbors; j++) {
                int neighbor = neighbors[i][j];
                double reachDist = Math.max(kDistances[neighbor], distances[i][neighbor]);
                sumReachDist += reachDist;
            }
            lrd[i] = nNeighbors / (sumReachDist + 1e-10);
        }
        
        // Compute LOF
        double[] lof = new double[n];
        for (int i = 0; i < n; i++) {
            double sumLrdRatio = 0;
            for (int j = 0; j < nNeighbors; j++) {
                int neighbor = neighbors[i][j];
                sumLrdRatio += lrd[neighbor] / (lrd[i] + 1e-10);
            }
            lof[i] = sumLrdRatio / nNeighbors;
        }
        
        return lof;
    }
    
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    private void sortByDistance(int[] indices, double[] distances) {
        for (int i = 0; i < indices.length - 1; i++) {
            for (int j = i + 1; j < indices.length; j++) {
                if (distances[j] < distances[i]) {
                    double tmpDist = distances[i];
                    distances[i] = distances[j];
                    distances[j] = tmpDist;
                    int tmpIdx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmpIdx;
                }
            }
        }
    }
    
    public static class Builder {
        private int nNeighbors = 20;
        private double contamination = 0.1;
        
        public Builder nNeighbors(int n) { this.nNeighbors = n; return this; }
        public Builder contamination(double c) { this.contamination = c; return this; }
        public LocalOutlierFactor build() { return new LocalOutlierFactor(nNeighbors, contamination); }
    }
}
