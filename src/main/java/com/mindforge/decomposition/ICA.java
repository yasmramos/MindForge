package com.mindforge.decomposition;

import java.util.Random;

/**
 * Independent Component Analysis (ICA) using FastICA algorithm.
 * 
 * ICA separates a multivariate signal into additive, independent components.
 * Unlike PCA which maximizes variance, ICA maximizes statistical independence.
 * 
 * Example usage:
 * <pre>
 * ICA ica = new ICA(3);
 * ica.fit(X);
 * double[][] sources = ica.transform(X);
 * </pre>
 * 
 * @author MindForge
 */
public class ICA {
    
    private final int nComponents;
    private final int maxIterations;
    private final double tolerance;
    private final Random random;
    
    private double[][] unmixingMatrix;
    private double[][] mixingMatrix;
    private double[] mean;
    private boolean fitted;
    
    public ICA(int nComponents) {
        this(nComponents, 200, 1e-4, new Random());
    }
    
    public ICA(int nComponents, int maxIterations, double tolerance, Random random) {
        this.nComponents = nComponents;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.random = random;
        this.fitted = false;
    }
    
    public void fit(double[][] X) {
        int n = X.length;
        int m = X[0].length;
        
        // Center data
        mean = new double[m];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                mean[j] += X[i][j];
            }
            mean[j] /= n;
        }
        
        double[][] centered = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                centered[i][j] = X[i][j] - mean[j];
            }
        }
        
        // Whiten data using PCA
        double[][] whitened = whiten(centered);
        
        // FastICA
        unmixingMatrix = fastICA(whitened);
        
        // Compute mixing matrix (pseudo-inverse)
        mixingMatrix = pseudoInverse(unmixingMatrix);
        
        fitted = true;
    }
    
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before transform");
        }
        
        int n = X.length;
        int m = X[0].length;
        
        // Center
        double[][] centered = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                centered[i][j] = X[i][j] - mean[j];
            }
        }
        
        // Whiten
        double[][] whitened = whiten(centered);
        
        // Apply unmixing
        double[][] result = new double[n][nComponents];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nComponents; j++) {
                for (int k = 0; k < whitened[0].length; k++) {
                    result[i][j] += unmixingMatrix[j][k] * whitened[i][k];
                }
            }
        }
        
        return result;
    }
    
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }
    
    private double[][] whiten(double[][] X) {
        int n = X.length;
        int m = X[0].length;
        int components = Math.min(nComponents, Math.min(n, m));
        
        // Covariance matrix
        double[][] cov = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += X[k][i] * X[k][j];
                }
                cov[i][j] = sum / (n - 1);
                cov[j][i] = cov[i][j];
            }
        }
        
        // Eigendecomposition (power iteration)
        double[][] eigenvectors = new double[m][components];
        double[] eigenvalues = new double[components];
        
        for (int c = 0; c < components; c++) {
            double[] v = new double[m];
            for (int i = 0; i < m; i++) v[i] = random.nextGaussian();
            normalize(v);
            
            for (int iter = 0; iter < 100; iter++) {
                double[] newV = new double[m];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < m; j++) {
                        newV[i] += cov[i][j] * v[j];
                    }
                }
                
                // Orthogonalize
                for (int prev = 0; prev < c; prev++) {
                    double dot = 0;
                    for (int i = 0; i < m; i++) dot += newV[i] * eigenvectors[i][prev];
                    for (int i = 0; i < m; i++) newV[i] -= dot * eigenvectors[i][prev];
                }
                
                double norm = 0;
                for (double val : newV) norm += val * val;
                norm = Math.sqrt(norm);
                
                if (norm > 1e-10) {
                    for (int i = 0; i < m; i++) newV[i] /= norm;
                }
                
                v = newV;
            }
            
            for (int i = 0; i < m; i++) eigenvectors[i][c] = v[i];
            
            double lambda = 0;
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int j = 0; j < m; j++) sum += cov[i][j] * v[j];
                lambda += v[i] * sum;
            }
            eigenvalues[c] = lambda;
        }
        
        // Whitening transformation
        double[][] whitened = new double[n][components];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < components; j++) {
                double sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += X[i][k] * eigenvectors[k][j];
                }
                whitened[i][j] = eigenvalues[j] > 1e-10 ? sum / Math.sqrt(eigenvalues[j]) : 0;
            }
        }
        
        return whitened;
    }
    
    private double[][] fastICA(double[][] X) {
        int n = X.length;
        int m = X[0].length;
        
        double[][] W = new double[nComponents][m];
        
        // Initialize W randomly
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < m; j++) {
                W[i][j] = random.nextGaussian();
            }
            normalize(W[i]);
        }
        
        for (int c = 0; c < nComponents; c++) {
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] wOld = W[c].clone();
                
                // w = E{X * g(w'X)} - E{g'(w'X)} * w
                double[] newW = new double[m];
                double gPrimeSum = 0;
                
                for (int i = 0; i < n; i++) {
                    double wx = 0;
                    for (int j = 0; j < m; j++) wx += W[c][j] * X[i][j];
                    
                    double g = Math.tanh(wx);
                    double gPrime = 1 - g * g;
                    
                    for (int j = 0; j < m; j++) {
                        newW[j] += X[i][j] * g;
                    }
                    gPrimeSum += gPrime;
                }
                
                for (int j = 0; j < m; j++) {
                    newW[j] = newW[j] / n - (gPrimeSum / n) * W[c][j];
                }
                
                // Decorrelation
                for (int prev = 0; prev < c; prev++) {
                    double dot = 0;
                    for (int j = 0; j < m; j++) dot += newW[j] * W[prev][j];
                    for (int j = 0; j < m; j++) newW[j] -= dot * W[prev][j];
                }
                
                normalize(newW);
                W[c] = newW;
                
                // Check convergence
                double diff = 0;
                for (int j = 0; j < m; j++) {
                    diff += Math.abs(Math.abs(W[c][j]) - Math.abs(wOld[j]));
                }
                if (diff < tolerance) break;
            }
        }
        
        return W;
    }
    
    private void normalize(double[] v) {
        double norm = 0;
        for (double val : v) norm += val * val;
        norm = Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < v.length; i++) v[i] /= norm;
        }
    }
    
    private double[][] pseudoInverse(double[][] W) {
        int rows = W.length;
        int cols = W[0].length;
        
        // W^T * (W * W^T)^-1 for fat matrices
        double[][] WWT = new double[rows][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < cols; k++) {
                    WWT[i][j] += W[i][k] * W[j][k];
                }
            }
        }
        
        // Simple inverse for small matrix
        double[][] inv = invertMatrix(WWT);
        
        double[][] result = new double[cols][rows];
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < rows; k++) {
                    result[i][j] += W[k][i] * inv[k][j];
                }
            }
        }
        
        return result;
    }
    
    private double[][] invertMatrix(double[][] A) {
        int n = A.length;
        double[][] aug = new double[n][2 * n];
        
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, aug[i], 0, n);
            aug[i][n + i] = 1;
        }
        
        for (int i = 0; i < n; i++) {
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(aug[k][i]) > Math.abs(aug[maxRow][i])) maxRow = k;
            }
            double[] temp = aug[i];
            aug[i] = aug[maxRow];
            aug[maxRow] = temp;
            
            if (Math.abs(aug[i][i]) < 1e-10) {
                aug[i][i] = 1e-10;
            }
            
            for (int k = i + 1; k < 2 * n; k++) {
                aug[i][k] /= aug[i][i];
            }
            aug[i][i] = 1;
            
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = aug[k][i];
                    for (int j = i; j < 2 * n; j++) {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }
        
        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(aug[i], n, inv[i], 0, n);
        }
        
        return inv;
    }
    
    public double[][] getUnmixingMatrix() { return unmixingMatrix; }
    public double[][] getMixingMatrix() { return mixingMatrix; }
    public int getNComponents() { return nComponents; }
}
