package io.github.yasmramos.mindforge.regression;

import java.io.Serializable;

/**
 * Elastic Net Regression (L1 + L2 Regularization).
 * 
 * Combines L1 and L2 penalties: alpha * l1_ratio * ||w||_1 + alpha * (1-l1_ratio) * ||w||^2
 * Uses coordinate descent optimization.
 * 
 * @author MindForge
 */
public class ElasticNet implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private double alpha;
    private double l1Ratio;
    private boolean fitIntercept;
    private int maxIterations;
    private double tolerance;
    
    private double[] coefficients;
    private double intercept;
    private boolean trained;
    private int nIterations;
    
    /**
     * Creates an Elastic Net model with default parameters.
     */
    public ElasticNet() {
        this(1.0, 0.5);
    }
    
    /**
     * Creates an Elastic Net model with specified alpha and l1_ratio.
     * 
     * @param alpha Overall regularization strength
     * @param l1Ratio Balance between L1 and L2 (0 = Ridge, 1 = Lasso)
     */
    public ElasticNet(double alpha, double l1Ratio) {
        this(alpha, l1Ratio, true, 1000, 1e-4);
    }
    
    /**
     * Creates an Elastic Net model with full configuration.
     */
    public ElasticNet(double alpha, double l1Ratio, boolean fitIntercept,
                      int maxIterations, double tolerance) {
        if (alpha < 0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        if (l1Ratio < 0 || l1Ratio > 1) {
            throw new IllegalArgumentException("l1Ratio must be between 0 and 1");
        }
        if (maxIterations <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive");
        }
        if (tolerance <= 0) {
            throw new IllegalArgumentException("tolerance must be positive");
        }
        
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.fitIntercept = fitIntercept;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.trained = false;
    }
    
    /**
     * Fits the model using coordinate descent.
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
        
        coefficients = new double[m];
        
        // Calculate means for centering
        double yMean = 0;
        double[] xMeans = new double[m];
        double[] xNorms = new double[m];
        
        for (int i = 0; i < n; i++) {
            yMean += y[i];
        }
        yMean /= n;
        
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                xMeans[j] += X[i][j];
            }
            xMeans[j] /= n;
            
            for (int i = 0; i < n; i++) {
                double centered = X[i][j] - (fitIntercept ? xMeans[j] : 0);
                xNorms[j] += centered * centered;
            }
        }
        
        // Center data
        double[][] XCentered = new double[n][m];
        double[] yCentered = new double[n];
        
        for (int i = 0; i < n; i++) {
            yCentered[i] = y[i] - (fitIntercept ? yMean : 0);
            for (int j = 0; j < m; j++) {
                XCentered[i][j] = X[i][j] - (fitIntercept ? xMeans[j] : 0);
            }
        }
        
        // Compute effective penalties
        double l1Penalty = alpha * l1Ratio * n;
        double l2Penalty = alpha * (1 - l1Ratio) * n;
        
        // Coordinate descent
        double[] residuals = yCentered.clone();
        
        for (nIterations = 0; nIterations < maxIterations; nIterations++) {
            double maxChange = 0;
            
            for (int j = 0; j < m; j++) {
                double denominator = xNorms[j] + l2Penalty;
                if (denominator == 0) continue;
                
                double oldCoef = coefficients[j];
                
                // Calculate partial residual
                double rho = 0;
                for (int i = 0; i < n; i++) {
                    rho += XCentered[i][j] * (residuals[i] + XCentered[i][j] * oldCoef);
                }
                
                // Soft thresholding with L2 adjustment
                coefficients[j] = softThreshold(rho, l1Penalty) / denominator;
                
                // Update residuals
                double coefChange = coefficients[j] - oldCoef;
                if (coefChange != 0) {
                    for (int i = 0; i < n; i++) {
                        residuals[i] -= XCentered[i][j] * coefChange;
                    }
                }
                
                maxChange = Math.max(maxChange, Math.abs(coefChange));
            }
            
            if (maxChange < tolerance) {
                break;
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
    
    private double softThreshold(double x, double lambda) {
        if (x > lambda) {
            return x - lambda;
        } else if (x < -lambda) {
            return x + lambda;
        } else {
            return 0;
        }
    }
    
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
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getL1Ratio() {
        return l1Ratio;
    }
    
    public boolean isTrained() {
        return trained;
    }
    
    public int getNIterations() {
        return nIterations;
    }
    
    public int getNumNonZeroCoefficients() {
        if (!trained) return 0;
        int count = 0;
        for (double coef : coefficients) {
            if (Math.abs(coef) > 1e-10) count++;
        }
        return count;
    }
}
