package io.github.yasmramos.mindforge.clustering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * OPTICS (Ordering Points To Identify the Clustering Structure) algorithm.
 * Density-based clustering that creates an augmented ordering of the database.
 */
public class OPTICS implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int minSamples;
    private final double maxEps;
    private final double xi;
    private int[] labels;
    private double[] reachabilityDistances;
    private int[] ordering;
    
    public OPTICS(int minSamples, double maxEps, double xi) {
        this.minSamples = minSamples;
        this.maxEps = maxEps;
        this.xi = xi;
    }
    
    public OPTICS() {
        this(5, Double.MAX_VALUE, 0.05);
    }
    
    public int[] fitPredict(double[][] X) {
        fit(X);
        return labels;
    }
    
    public void fit(double[][] X) {
        int n = X.length;
        reachabilityDistances = new double[n];
        Arrays.fill(reachabilityDistances, Double.MAX_VALUE);
        boolean[] processed = new boolean[n];
        ordering = new int[n];
        int orderIdx = 0;
        
        for (int i = 0; i < n; i++) {
            if (processed[i]) continue;
            
            List<int[]> neighbors = getNeighbors(X, i, maxEps);
            processed[i] = true;
            ordering[orderIdx++] = i;
            
            if (coreDistance(X, i, neighbors) != Double.MAX_VALUE) {
                PriorityQueue<int[]> seeds = new PriorityQueue<>((a, b) -> 
                    Double.compare(reachabilityDistances[a[0]], reachabilityDistances[b[0]]));
                update(X, i, neighbors, seeds, processed);
                
                while (!seeds.isEmpty()) {
                    int[] current = seeds.poll();
                    int currentIdx = current[0];
                    if (processed[currentIdx]) continue;
                    
                    List<int[]> currentNeighbors = getNeighbors(X, currentIdx, maxEps);
                    processed[currentIdx] = true;
                    ordering[orderIdx++] = currentIdx;
                    
                    if (coreDistance(X, currentIdx, currentNeighbors) != Double.MAX_VALUE) {
                        update(X, currentIdx, currentNeighbors, seeds, processed);
                    }
                }
            }
        }
        
        // Extract clusters using xi method
        extractClusters(X);
    }
    
    private void update(double[][] X, int p, List<int[]> neighbors, 
                       PriorityQueue<int[]> seeds, boolean[] processed) {
        double coreDist = coreDistance(X, p, neighbors);
        
        for (int[] neighbor : neighbors) {
            int o = neighbor[0];
            if (processed[o]) continue;
            
            double newReachDist = Math.max(coreDist, distance(X[p], X[o]));
            if (reachabilityDistances[o] == Double.MAX_VALUE) {
                reachabilityDistances[o] = newReachDist;
                seeds.add(new int[]{o});
            } else if (newReachDist < reachabilityDistances[o]) {
                reachabilityDistances[o] = newReachDist;
                seeds.add(new int[]{o});
            }
        }
    }
    
    private double coreDistance(double[][] X, int p, List<int[]> neighbors) {
        if (neighbors.size() < minSamples) {
            return Double.MAX_VALUE;
        }
        double[] distances = new double[neighbors.size()];
        for (int i = 0; i < neighbors.size(); i++) {
            distances[i] = distance(X[p], X[neighbors.get(i)[0]]);
        }
        Arrays.sort(distances);
        return distances[minSamples - 1];
    }
    
    private List<int[]> getNeighbors(double[][] X, int p, double eps) {
        List<int[]> neighbors = new ArrayList<>();
        for (int i = 0; i < X.length; i++) {
            double dist = distance(X[p], X[i]);
            if (dist <= eps) {
                neighbors.add(new int[]{i});
            }
        }
        return neighbors;
    }
    
    private void extractClusters(double[][] X) {
        labels = new int[X.length];
        Arrays.fill(labels, -1);
        
        int currentCluster = 0;
        boolean inCluster = false;
        
        for (int i = 1; i < ordering.length; i++) {
            int idx = ordering[i];
            int prevIdx = ordering[i - 1];
            
            double reach = reachabilityDistances[idx];
            double prevReach = reachabilityDistances[prevIdx];
            
            if (reach == Double.MAX_VALUE) {
                inCluster = false;
                continue;
            }
            
            if (!inCluster && reach < maxEps) {
                inCluster = true;
                currentCluster++;
            }
            
            if (inCluster) {
                if (prevReach != Double.MAX_VALUE && 
                    reach > prevReach * (1 + xi)) {
                    currentCluster++;
                }
                labels[idx] = currentCluster;
            }
        }
    }
    
    private double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    public int[] getLabels() { return labels; }
    public double[] getReachabilityDistances() { return reachabilityDistances; }
    public int[] getOrdering() { return ordering; }
    
    public static class Builder {
        private int minSamples = 5;
        private double maxEps = Double.MAX_VALUE;
        private double xi = 0.05;
        
        public Builder minSamples(int n) { this.minSamples = n; return this; }
        public Builder maxEps(double e) { this.maxEps = e; return this; }
        public Builder xi(double x) { this.xi = x; return this; }
        public OPTICS build() { return new OPTICS(minSamples, maxEps, xi); }
    }
}
