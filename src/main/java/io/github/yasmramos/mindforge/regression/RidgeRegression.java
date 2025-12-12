package io.github.yasmramos.mindforge.regression;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Ridge Regression (L2 Regularization).
 * 
 * Ridge regression adds L2 penalty to the loss function to prevent overfitting.
 * The objective is to minimize: ||y - Xw||^2 + alpha * ||w||^2
 * 
 * @author MindForge
 */
public class RidgeRegression implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private double alpha;
    private boolean fitIntercept;
    private boolean normalize;
    private int maxIterations;
    private double tolerance;
    
    private double[] coefficients;
    private double intercept;
    private boolean trained;
    
    /**
     * Creates a Ridge Regression model with default parameters.
     */
    public RidgeRegression() {
        this(1.0);
    }
    
    /**
     * Creates a Ridge Regression model with specified alpha.
     * 
     * @param alpha Regularization strength (must be positive)
     */
    public RidgeRegression(double alpha) {
        this(alpha, true, false, 1000, 1e-6);
    }
    
    /**
     * Creates a Ridge Regression model with full configuration.
     * 
     * @param alpha Regularization strength
     * @param fitIntercept Whether to fit intercept
     * @param normalize Whether to normalize features
     * @param maxIterations Maximum iterations for optimization
     * @param tolerance Convergence tolerance
     */
    public RidgeRegression(double alpha, boolean fitIntercept, boolean normalize, 
                           int maxIterations, double tolerance) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive");
        }
        if (tolerance <= 0) {
            throw new IllegalArgumentException("tolerance must be positive");
        }
        
        this.alpha = alpha;
        this.fitIntercept = fitIntercept;
        this.normalize = normalize;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.trained = false;
    }
    
    /**
     * Fits the model using closed-form solution.
     * w = (X^T X + alpha * I)^(-1) X^T y
     * 
     * @param X Feature matrix
     * @param y Target values
     */
    public void train(double[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same number of samples");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        
        int n = X.length;
        int m = X[0].length;
        
        // Calculate mean for intercept
        double yMean = 0;
        double[] xMeans = new double[m];
        double[] xStds = new double[m];
        
        if (fitIntercept) {
            for (int i = 0; i < n; i++) {
                yMean += y[i];
            }
            yMean /= n;
            
            for (int j = 0; j < m; j++) {
                for (int i = 0; i < n; i++) {
                    xMeans[j] += X[i][j];
                }
                xMeans[j] /= n;
            }
        }
        
        if (normalize) {
            for (int j = 0; j < m; j++) {
                double sumSq = 0;
                for (int i = 0; i < n; i++) {
                    double diff = X[i][j] - xMeans[j];
                    sumSq += diff * diff;
                }
                xStds[j] = Math.sqrt(sumSq / n);
                if (xStds[j] == 0) xStds[j] = 1;
            }
        } else {
            Arrays.fill(xStds, 1.0);
        }
        
        // Center and normalize data
        double[][] XCentered = new double[n][m];
        double[] yCentered = new double[n];
        
        for (int i = 0; i < n; i++) {
            yCentered[i] = y[i] - (fitIntercept ? yMean : 0);
            for (int j = 0; j < m; j++) {
                XCentered[i][j] = (X[i][j] - (fitIntercept ? xMeans[j] : 0)) / xStds[j];
            }
        }
        
        // Compute X^T X + alpha * I
        double[][] XtX = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < n; k++) {
                    XtX[i][j] += XCentered[k][i] * XCentered[k][j];
                }
            }
            XtX[i][i] += alpha; // Add regularization
        }
        
        // Compute X^T y
        double[] Xty = new double[m];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                Xty[i] += XCentered[k][i] * yCentered[k];
            }
        }
        
        // Solve using Cholesky or gradient descent
        coefficients = solveLinearSystem(XtX, Xty);
        
        // Rescale coefficients if normalized
        if (normalize) {
            for (int j = 0; j < m; j++) {
                coefficients[j] /= xStds[j];
            }
        }
        
        // Calculate intercept
        if (fitIntercept) {
            intercept = yMean;
            for (int j = 0; j < m; j++) {
                intercept -= coefficients[j] * xMeans[j];
            }
        } else {
            intercept = 0;
        }
        
        trained = true;
    }
    
    /**
     * Solves Ax = b using Gaussian elimination with partial pivoting.
     */
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = b.length;
        double[][] augmented = new double[n][n + 1];
        
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, augmented[i], 0, n);
            augmented[i][n] = b[i];
        }
        
        // Forward elimination
        for (int k = 0; k < n; k++) {
            // Partial pivoting
            int maxRow = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(augmented[i][k]) > Math.abs(augmented[maxRow][k])) {
                    maxRow = i;
                }
            }
            double[] temp = augmented[k];
            augmented[k] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            if (Math.abs(augmented[k][k]) < 1e-10) {
                continue; // Skip near-zero pivots
            }
            
            for (int i = k + 1; i < n; i++) {
                double factor = augmented[i][k] / augmented[k][k];
                for (int j = k; j <= n; j++) {
                    augmented[i][j] -= factor * augmented[k][j];
                }
            }
        }
        
        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            if (Math.abs(augmented[i][i]) > 1e-10) {
                x[i] /= augmented[i][i];
            }
        }
        
        return x;
    }
    
    /**
     * Predicts target value for a single sample.
     */
    public double predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        if (x == null || x.length != coefficients.length) {
            throw new IllegalArgumentException("Invalid input dimensions");
        }
        
        double prediction = intercept;
        for (int i = 0; i < x.length; i++) {
            prediction += coefficients[i] * x[i];
        }
        return prediction;
    }
    
    /**
     * Predicts target values for multiple samples.
     */
    public double[] predict(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Computes R² score.
     */
    public double score(double[][] X, double[] y) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        double[] predictions = predict(X);
        
        double yMean = 0;
        for (double val : y) yMean += val;
        yMean /= y.length;
        
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < y.length; i++) {
            ssRes += Math.pow(y[i] - predictions[i], 2);
            ssTot += Math.pow(y[i] - yMean, 2);
        }
        
        return ssTot == 0 ? 0 : 1 - (ssRes / ssTot);
    }
    
    // Getters
    public double[] getCoefficients() {
        return coefficients != null ? coefficients.clone() : null;
    }
    
    public double getIntercept() {
        return intercept;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public boolean isTrained() {
        return trained;
    }
    
    public boolean isFitIntercept() {
        return fitIntercept;
    }
    
    public boolean isNormalize() {
        return normalize;
    }
}
