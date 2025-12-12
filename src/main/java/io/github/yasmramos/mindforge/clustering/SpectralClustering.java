package io.github.yasmramos.mindforge.clustering;

import java.util.Random;

/**
 * Spectral Clustering implementation.
 * Uses graph Laplacian eigenvectors for clustering.
 */
public class SpectralClustering implements Clusterer<double[]> {
    
    private final int k;
    private final double gamma;
    private final int maxIterKMeans;
    private final Random random;
    
    private int[] labels;
    private double[][] embedding;
    private double[][] centroids;
    private boolean fitted;
    
    public SpectralClustering(int k) {
        this(k, 1.0, 300, new Random());
    }
    
    public SpectralClustering(int k, double gamma, int maxIterKMeans, Random random) {
        this.k = k;
        this.gamma = gamma;
        this.maxIterKMeans = maxIterKMeans;
        this.random = random;
        this.fitted = false;
    }
    
    @Override
    public int[] cluster(double[][] x) {
        int n = x.length;
        
        double[][] affinity = buildAffinityMatrix(x);
        double[][] laplacian = computeNormalizedLaplacian(affinity, n);
        double[][] eigenvectors = computeEigenvectors(laplacian, k);
        normalizeRows(eigenvectors);
        
        this.embedding = eigenvectors;
        this.labels = kMeansClustering(eigenvectors, k);
        this.fitted = true;
        
        return labels;
    }
    
    @Override
    public int predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        int nearest = 0;
        double minDist = Double.MAX_VALUE;
        for (int c = 0; c < k; c++) {
            double dist = 0;
            for (int j = 0; j < centroids[c].length; j++) {
                double diff = x[j] - centroids[c][j];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                nearest = c;
            }
        }
        return nearest;
    }
    
    @Override
    public int getNumClusters() { return k; }
    
    public int[] getLabels() { return labels; }
    
    public double[][] getEmbedding() { return embedding; }
    
    private double[][] buildAffinityMatrix(double[][] X) {
        int n = X.length;
        double[][] affinity = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    affinity[i][j] = 0;
                } else {
                    double dist = squaredDistance(X, i, j);
                    double sim = Math.exp(-gamma * dist);
                    affinity[i][j] = sim;
                    affinity[j][i] = sim;
                }
            }
        }
        return affinity;
    }
    
    private double squaredDistance(double[][] X, int i, int j) {
        double sum = 0;
        int d = X[0].length;
        for (int dim = 0; dim < d; dim++) {
            double diff = X[i][dim] - X[j][dim];
            sum += diff * diff;
        }
        return sum;
    }
    
    private double[][] computeNormalizedLaplacian(double[][] affinity, int n) {
        double[] degree = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                degree[i] += affinity[i][j];
            }
        }
        
        double[] dInvSqrt = new double[n];
        for (int i = 0; i < n; i++) {
            dInvSqrt[i] = degree[i] > 0 ? 1.0 / Math.sqrt(degree[i]) : 0;
        }
        
        double[][] laplacian = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    laplacian[i][j] = 1.0 - dInvSqrt[i] * affinity[i][j] * dInvSqrt[j];
                } else {
                    laplacian[i][j] = -dInvSqrt[i] * affinity[i][j] * dInvSqrt[j];
                }
            }
        }
        return laplacian;
    }
    
    private double[][] computeEigenvectors(double[][] matrix, int numVectors) {
        int n = matrix.length;
        double[][] eigenvectors = new double[n][numVectors];
        
        double[][] shifted = new double[n][n];
        double shift = 2.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                shifted[i][j] = (i == j) ? shift - matrix[i][j] : -matrix[i][j];
            }
        }
        
        for (int ev = 0; ev < numVectors; ev++) {
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = random.nextGaussian();
            }
            normalize(v);
            
            for (int prev = 0; prev < ev; prev++) {
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += v[i] * eigenvectors[i][prev];
                }
                for (int i = 0; i < n; i++) {
                    v[i] -= dot * eigenvectors[i][prev];
                }
                normalize(v);
            }
            
            for (int iter = 0; iter < 100; iter++) {
                double[] newV = new double[n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        newV[i] += shifted[i][j] * v[j];
                    }
                }
                
                for (int prev = 0; prev < ev; prev++) {
                    double dot = 0;
                    for (int i = 0; i < n; i++) {
                        dot += newV[i] * eigenvectors[i][prev];
                    }
                    for (int i = 0; i < n; i++) {
                        newV[i] -= dot * eigenvectors[i][prev];
                    }
                }
                
                normalize(newV);
                
                double diff = 0;
                for (int i = 0; i < n; i++) {
                    diff += (newV[i] - v[i]) * (newV[i] - v[i]);
                }
                v = newV;
                
                if (Math.sqrt(diff) < 1e-10) break;
            }
            
            for (int i = 0; i < n; i++) {
                eigenvectors[i][ev] = v[i];
            }
        }
        
        return eigenvectors;
    }
    
    private void normalize(double[] v) {
        double norm = 0;
        for (double val : v) norm += val * val;
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < v.length; i++) v[i] /= norm;
        }
    }
    
    private void normalizeRows(double[][] matrix) {
        for (double[] row : matrix) {
            double norm = 0;
            for (double val : row) norm += val * val;
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < row.length; i++) row[i] /= norm;
            }
        }
    }
    
    private int[] kMeansClustering(double[][] data, int numClusters) {
        int n = data.length;
        int d = data[0].length;
        
        centroids = new double[numClusters][d];
        int[] assignments = new int[n];
        
        for (int i = 0; i < numClusters; i++) {
            int idx = random.nextInt(n);
            System.arraycopy(data[idx], 0, centroids[i], 0, d);
        }
        
        for (int iter = 0; iter < maxIterKMeans; iter++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int nearest = 0;
                double minDist = Double.MAX_VALUE;
                for (int c = 0; c < numClusters; c++) {
                    double dist = 0;
                    for (int j = 0; j < d; j++) {
                        double diff = data[i][j] - centroids[c][j];
                        dist += diff * diff;
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        nearest = c;
                    }
                }
                if (assignments[i] != nearest) {
                    assignments[i] = nearest;
                    changed = true;
                }
            }
            
            if (!changed) break;
            
            int[] counts = new int[numClusters];
            double[][] newCentroids = new double[numClusters][d];
            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                counts[c]++;
                for (int j = 0; j < d; j++) {
                    newCentroids[c][j] += data[i][j];
                }
            }
            for (int c = 0; c < numClusters; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < d; j++) {
                        centroids[c][j] = newCentroids[c][j] / counts[c];
                    }
                }
            }
        }
        
        return assignments;
    }
}
