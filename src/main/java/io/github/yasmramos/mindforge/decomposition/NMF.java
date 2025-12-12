package io.github.yasmramos.mindforge.decomposition;

import java.util.Random;

/**
 * Non-negative Matrix Factorization (NMF).
 * 
 * Decomposes a non-negative matrix V into W * H where W and H are non-negative.
 * Useful for topic modeling, image processing, and feature extraction.
 * 
 * Uses multiplicative update rules for optimization.
 * 
 * @author MindForge
 */
public class NMF {
    
    private final int nComponents;
    private final int maxIterations;
    private final double tolerance;
    private final Random random;
    
    private double[][] W;
    private double[][] H;
    private boolean fitted;
    
    public NMF(int nComponents) {
        this(nComponents, 200, 1e-4, new Random());
    }
    
    public NMF(int nComponents, int maxIterations, double tolerance, Random random) {
        this.nComponents = nComponents;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.random = random;
        this.fitted = false;
    }
    
    public void fit(double[][] V) {
        int n = V.length;
        int m = V[0].length;
        
        // Initialize W and H with small random positive values
        W = new double[n][nComponents];
        H = new double[nComponents][m];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nComponents; j++) {
                W[i][j] = random.nextDouble() * 0.1 + 0.01;
            }
        }
        for (int i = 0; i < nComponents; i++) {
            for (int j = 0; j < m; j++) {
                H[i][j] = random.nextDouble() * 0.1 + 0.01;
            }
        }
        
        double prevError = Double.MAX_VALUE;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            // Update H: H = H * (W^T * V) / (W^T * W * H)
            double[][] WtV = multiply(transpose(W), V);
            double[][] WtW = multiply(transpose(W), W);
            double[][] WtWH = multiply(WtW, H);
            
            for (int i = 0; i < nComponents; i++) {
                for (int j = 0; j < m; j++) {
                    if (WtWH[i][j] > 1e-10) {
                        H[i][j] *= WtV[i][j] / WtWH[i][j];
                    }
                    H[i][j] = Math.max(H[i][j], 1e-10);
                }
            }
            
            // Update W: W = W * (V * H^T) / (W * H * H^T)
            double[][] VHt = multiply(V, transpose(H));
            double[][] HHt = multiply(H, transpose(H));
            double[][] WHHt = multiply(W, HHt);
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < nComponents; j++) {
                    if (WHHt[i][j] > 1e-10) {
                        W[i][j] *= VHt[i][j] / WHHt[i][j];
                    }
                    W[i][j] = Math.max(W[i][j], 1e-10);
                }
            }
            
            // Check convergence
            double error = reconstructionError(V);
            if (Math.abs(prevError - error) < tolerance) {
                break;
            }
            prevError = error;
        }
        
        fitted = true;
    }
    
    public double[][] transform(double[][] V) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted before transform");
        }
        
        int n = V.length;
        int m = V[0].length;
        
        // Solve for W_new given fixed H: W_new = V * H^T * (H * H^T)^-1
        double[][] Wnew = new double[n][nComponents];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nComponents; j++) {
                Wnew[i][j] = random.nextDouble() * 0.1 + 0.01;
            }
        }
        
        for (int iter = 0; iter < 100; iter++) {
            double[][] VHt = multiply(V, transpose(H));
            double[][] HHt = multiply(H, transpose(H));
            double[][] WnewHHt = multiply(Wnew, HHt);
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < nComponents; j++) {
                    if (WnewHHt[i][j] > 1e-10) {
                        Wnew[i][j] *= VHt[i][j] / WnewHHt[i][j];
                    }
                    Wnew[i][j] = Math.max(Wnew[i][j], 1e-10);
                }
            }
        }
        
        return Wnew;
    }
    
    public double[][] fitTransform(double[][] V) {
        fit(V);
        return W;
    }
    
    public double[][] inverseTransform(double[][] Wnew) {
        if (!fitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return multiply(Wnew, H);
    }
    
    private double reconstructionError(double[][] V) {
        double[][] WH = multiply(W, H);
        double error = 0;
        for (int i = 0; i < V.length; i++) {
            for (int j = 0; j < V[0].length; j++) {
                double diff = V[i][j] - WH[i][j];
                error += diff * diff;
            }
        }
        return Math.sqrt(error);
    }
    
    private double[][] multiply(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int k = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
        return C;
    }
    
    private double[][] transpose(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double[][] T = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }
    
    public double[][] getW() { return W; }
    public double[][] getH() { return H; }
    public int getNComponents() { return nComponents; }
}
