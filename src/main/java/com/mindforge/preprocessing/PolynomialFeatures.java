package com.mindforge.preprocessing;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Generate polynomial and interaction features.
 * 
 * <p>Generate a new feature matrix consisting of all polynomial combinations
 * of the features with degree less than or equal to the specified degree.</p>
 * 
 * <p>For example, if an input sample is two dimensional and of the form [a, b],
 * the degree-2 polynomial features are [1, a, b, a², ab, b²].</p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * PolynomialFeatures poly = new PolynomialFeatures(2);
 * double[][] X = {{2, 3}, {4, 5}};
 * double[][] X_poly = poly.fitTransform(X);
 * // X_poly = [[1, 2, 3, 4, 6, 9], [1, 4, 5, 16, 20, 25]]
 * }</pre>
 * 
 * @author Matrix Agent
 * @version 1.0
 */
public class PolynomialFeatures implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final int degree;
    private final boolean includeBias;
    private final boolean interactionOnly;
    private int nInputFeatures;
    private int nOutputFeatures;
    private boolean isFitted;
    private List<int[]> powers;
    
    /**
     * Creates a PolynomialFeatures transformer with specified degree.
     * 
     * @param degree The degree of the polynomial features. Default is 2.
     * @throws IllegalArgumentException if degree is less than 1
     */
    public PolynomialFeatures(int degree) {
        this(degree, true, false);
    }
    
    /**
     * Creates a PolynomialFeatures transformer with full configuration.
     * 
     * @param degree The degree of the polynomial features
     * @param includeBias If true (default), include a bias column (all ones)
     * @param interactionOnly If true, only interaction features are produced
     * @throws IllegalArgumentException if degree is less than 1
     */
    public PolynomialFeatures(int degree, boolean includeBias, boolean interactionOnly) {
        if (degree < 1) {
            throw new IllegalArgumentException("Degree must be at least 1, got: " + degree);
        }
        this.degree = degree;
        this.includeBias = includeBias;
        this.interactionOnly = interactionOnly;
        this.isFitted = false;
    }
    
    /**
     * Compute the number of output features and the power combinations.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return this transformer
     * @throws IllegalArgumentException if X is null or empty
     */
    public PolynomialFeatures fit(double[][] X) {
        validateInput(X);
        
        this.nInputFeatures = X[0].length;
        this.powers = generatePowers(nInputFeatures, degree, interactionOnly, includeBias);
        this.nOutputFeatures = powers.size();
        this.isFitted = true;
        
        return this;
    }
    
    /**
     * Transform data to polynomial features.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return Transformed data of shape [n_samples, n_output_features]
     * @throws IllegalStateException if the transformer has not been fitted
     * @throws IllegalArgumentException if X has different number of features than training data
     */
    public double[][] transform(double[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("PolynomialFeatures must be fitted before transform");
        }
        validateInput(X);
        
        if (X[0].length != nInputFeatures) {
            throw new IllegalArgumentException(
                String.format("X has %d features, but PolynomialFeatures is expecting %d features",
                    X[0].length, nInputFeatures));
        }
        
        int nSamples = X.length;
        double[][] result = new double[nSamples][nOutputFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nOutputFeatures; j++) {
                result[i][j] = computeFeature(X[i], powers.get(j));
            }
        }
        
        return result;
    }
    
    /**
     * Fit to data, then transform it.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return Transformed data of shape [n_samples, n_output_features]
     */
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }
    
    /**
     * Compute a single polynomial feature for a sample.
     */
    private double computeFeature(double[] sample, int[] power) {
        double result = 1.0;
        for (int i = 0; i < power.length; i++) {
            result *= Math.pow(sample[i], power[i]);
        }
        return result;
    }
    
    /**
     * Generate all power combinations for polynomial features.
     */
    private List<int[]> generatePowers(int nFeatures, int degree, boolean interactionOnly, boolean includeBias) {
        List<int[]> result = new ArrayList<>();
        
        // Start from 0 (bias term) or 1
        int startDegree = includeBias ? 0 : 1;
        
        for (int d = startDegree; d <= degree; d++) {
            generatePowersForDegree(result, new int[nFeatures], 0, d, interactionOnly);
        }
        
        return result;
    }
    
    /**
     * Recursively generate power combinations for a specific degree.
     */
    private void generatePowersForDegree(List<int[]> result, int[] current, int index, int remaining, boolean interactionOnly) {
        if (index == current.length) {
            if (remaining == 0) {
                result.add(current.clone());
            }
            return;
        }
        
        int maxPower = interactionOnly ? Math.min(1, remaining) : remaining;
        
        for (int p = 0; p <= maxPower; p++) {
            current[index] = p;
            generatePowersForDegree(result, current, index + 1, remaining - p, interactionOnly);
        }
        current[index] = 0;
    }
    
    /**
     * Validate input data.
     */
    private void validateInput(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        if (X[0] == null || X[0].length == 0) {
            throw new IllegalArgumentException("Input features cannot be null or empty");
        }
    }
    
    /**
     * Get the degree of polynomial features.
     * 
     * @return the degree
     */
    public int getDegree() {
        return degree;
    }
    
    /**
     * Check if bias column is included.
     * 
     * @return true if bias is included
     */
    public boolean isIncludeBias() {
        return includeBias;
    }
    
    /**
     * Check if only interaction features are generated.
     * 
     * @return true if interaction only mode
     */
    public boolean isInteractionOnly() {
        return interactionOnly;
    }
    
    /**
     * Get the number of input features.
     * 
     * @return number of input features
     * @throws IllegalStateException if not fitted
     */
    public int getNInputFeatures() {
        if (!isFitted) {
            throw new IllegalStateException("PolynomialFeatures must be fitted first");
        }
        return nInputFeatures;
    }
    
    /**
     * Get the number of output features.
     * 
     * @return number of output features after transformation
     * @throws IllegalStateException if not fitted
     */
    public int getNOutputFeatures() {
        if (!isFitted) {
            throw new IllegalStateException("PolynomialFeatures must be fitted first");
        }
        return nOutputFeatures;
    }
    
    /**
     * Check if the transformer has been fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return isFitted;
    }
}
