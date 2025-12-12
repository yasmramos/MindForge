package io.github.yasmramos.mindforge.regression;

import java.io.Serializable;

/**
 * Polynomial Regression.
 * 
 * Fits a polynomial model by transforming features to polynomial form
 * and applying linear regression.
 * 
 * @author MindForge
 */
public class PolynomialRegression implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private int degree;
    private boolean fitIntercept;
    private boolean includeBias;
    
    private double[] coefficients;
    private double intercept;
    private boolean trained;
    private int nFeatures;
    
    /**
     * Creates a Polynomial Regression with default degree 2.
     */
    public PolynomialRegression() {
        this(2);
    }
    
    /**
     * Creates a Polynomial Regression with specified degree.
     */
    public PolynomialRegression(int degree) {
        this(degree, true, false);
    }
    
    /**
     * Creates a Polynomial Regression with full configuration.
     * 
     * @param degree Polynomial degree
     * @param fitIntercept Whether to fit intercept
     * @param includeBias Whether to include bias term in features
     */
    public PolynomialRegression(int degree, boolean fitIntercept, boolean includeBias) {
        if (degree < 1) {
            throw new IllegalArgumentException("Degree must be at least 1");
        }
        if (degree > 10) {
            throw new IllegalArgumentException("Degree too high (max 10)");
        }
        
        this.degree = degree;
        this.fitIntercept = fitIntercept;
        this.includeBias = includeBias;
        this.trained = false;
    }
    
    /**
     * Fits the polynomial regression model.
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
        nFeatures = X[0].length;
        
        // Transform features to polynomial
        double[][] XPoly = transformFeatures(X);
        int m = XPoly[0].length;
        
        // Calculate means for centering
        double yMean = 0;
        double[] xMeans = new double[m];
        
        for (int i = 0; i < n; i++) {
            yMean += y[i];
        }
        yMean /= n;
        
        if (fitIntercept) {
            for (int j = 0; j < m; j++) {
                for (int i = 0; i < n; i++) {
                    xMeans[j] += XPoly[i][j];
                }
                xMeans[j] /= n;
            }
        }
        
        // Center data
        double[][] XCentered = new double[n][m];
        double[] yCentered = new double[n];
        
        for (int i = 0; i < n; i++) {
            yCentered[i] = y[i] - (fitIntercept ? yMean : 0);
            for (int j = 0; j < m; j++) {
                XCentered[i][j] = XPoly[i][j] - (fitIntercept ? xMeans[j] : 0);
            }
        }
        
        // Compute X^T X with small regularization for stability
        double[][] XtX = new double[m][m];
        double regularization = 1e-8;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < n; k++) {
                    XtX[i][j] += XCentered[k][i] * XCentered[k][j];
                }
            }
            XtX[i][i] += regularization;
        }
        
        // Compute X^T y
        double[] Xty = new double[m];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                Xty[i] += XCentered[k][i] * yCentered[k];
            }
        }
        
        // Solve linear system
        coefficients = solveLinearSystem(XtX, Xty);
        
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
     * Transforms features to polynomial form.
     */
    private double[][] transformFeatures(double[][] X) {
        int n = X.length;
        int origFeatures = X[0].length;
        
        // Calculate number of polynomial features
        int numFeatures = calculateNumFeatures(origFeatures, degree);
        if (includeBias) numFeatures++;
        
        double[][] XPoly = new double[n][numFeatures];
        
        for (int i = 0; i < n; i++) {
            int idx = 0;
            
            if (includeBias) {
                XPoly[i][idx++] = 1.0;
            }
            
            // Generate polynomial features
            for (int d = 1; d <= degree; d++) {
                if (origFeatures == 1) {
                    XPoly[i][idx++] = Math.pow(X[i][0], d);
                } else {
                    // For multi-feature, add all combinations up to degree d
                    idx = addPolynomialTerms(XPoly[i], idx, X[i], d);
                }
            }
        }
        
        return XPoly;
    }
    
    private int calculateNumFeatures(int n, int d) {
        if (n == 1) {
            return d;
        }
        // Simplified: for multi-feature, use sum of features for each degree
        int count = 0;
        for (int deg = 1; deg <= d; deg++) {
            count += (int) Math.pow(n, deg);
        }
        return Math.min(count, n * d + (n * (n-1)) / 2 * (d > 1 ? 1 : 0));
    }
    
    private int addPolynomialTerms(double[] result, int startIdx, double[] x, int targetDegree) {
        int idx = startIdx;
        int n = x.length;
        
        if (targetDegree == 1) {
            for (int i = 0; i < n; i++) {
                result[idx++] = x[i];
            }
        } else if (targetDegree == 2) {
            // x_i^2 and x_i * x_j
            for (int i = 0; i < n; i++) {
                result[idx++] = x[i] * x[i];
            }
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    result[idx++] = x[i] * x[j];
                }
            }
        } else {
            // Higher degrees: just powers
            for (int i = 0; i < n; i++) {
                result[idx++] = Math.pow(x[i], targetDegree);
            }
        }
        
        return idx;
    }
    
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = b.length;
        double[][] augmented = new double[n][n + 1];
        
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, augmented[i], 0, n);
            augmented[i][n] = b[i];
        }
        
        // Forward elimination with partial pivoting
        for (int k = 0; k < n; k++) {
            int maxRow = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(augmented[i][k]) > Math.abs(augmented[maxRow][k])) {
                    maxRow = i;
                }
            }
            
            double[] temp = augmented[k];
            augmented[k] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            if (Math.abs(augmented[k][k]) < 1e-10) continue;
            
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
    
    public double predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        if (x == null || x.length != nFeatures) {
            throw new IllegalArgumentException("Invalid input dimensions");
        }
        
        double[][] xMatrix = {x};
        double[][] xPoly = transformFeatures(xMatrix);
        
        double prediction = intercept;
        for (int i = 0; i < coefficients.length; i++) {
            prediction += coefficients[i] * xPoly[0][i];
        }
        return prediction;
    }
    
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
    
    public int getDegree() {
        return degree;
    }
    
    public boolean isTrained() {
        return trained;
    }
    
    public int getNumCoefficients() {
        return coefficients != null ? coefficients.length : 0;
    }
}
