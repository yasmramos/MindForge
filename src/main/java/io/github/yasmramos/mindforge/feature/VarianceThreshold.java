package io.github.yasmramos.mindforge.feature;

import java.util.ArrayList;
import java.util.List;

/**
 * Feature selector that removes all low-variance features.
 * 
 * This feature selection algorithm looks only at X, not y, and thus
 * can be used for unsupervised learning.
 * 
 * Features with a variance lower than the threshold will be removed.
 * By default, it removes all zero-variance features (constant features).
 * 
 * Example usage:
 * <pre>
 * VarianceThreshold selector = new VarianceThreshold(0.1);
 * selector.fit(X);
 * double[][] X_selected = selector.transform(X);
 * // Or in one step:
 * double[][] X_selected = selector.fitTransform(X);
 * </pre>
 * 
 * @author MindForge
 * @version 1.0.8-alpha
 */
public class VarianceThreshold {
    
    private double threshold;
    private double[] variances;
    private int[] selectedFeatureIndices;
    private boolean fitted;
    
    /**
     * Creates a VarianceThreshold selector with default threshold of 0.0.
     * This will remove only constant features (zero variance).
     */
    public VarianceThreshold() {
        this(0.0);
    }
    
    /**
     * Creates a VarianceThreshold selector with specified threshold.
     * 
     * @param threshold Features with variance lower than this will be removed.
     *                  Must be non-negative.
     * @throws IllegalArgumentException if threshold is negative
     */
    public VarianceThreshold(double threshold) {
        if (threshold < 0) {
            throw new IllegalArgumentException("Threshold must be non-negative, got: " + threshold);
        }
        this.threshold = threshold;
        this.fitted = false;
    }
    
    /**
     * Computes variances of features and identifies which to keep.
     * 
     * @param X Training data of shape [n_samples, n_features]
     * @return this selector for method chaining
     * @throws IllegalArgumentException if X is null or empty
     */
    public VarianceThreshold fit(double[][] X) {
        validateInput(X);
        
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Calculate variance for each feature
        variances = new double[nFeatures];
        List<Integer> selectedIndices = new ArrayList<>();
        
        for (int j = 0; j < nFeatures; j++) {
            // Calculate mean
            double mean = 0.0;
            for (int i = 0; i < nSamples; i++) {
                mean += X[i][j];
            }
            mean /= nSamples;
            
            // Calculate variance
            double variance = 0.0;
            for (int i = 0; i < nSamples; i++) {
                double diff = X[i][j] - mean;
                variance += diff * diff;
            }
            variance /= nSamples;
            variances[j] = variance;
            
            // Keep feature if variance is above threshold
            if (variance > threshold) {
                selectedIndices.add(j);
            }
        }
        
        // Convert to array
        selectedFeatureIndices = selectedIndices.stream().mapToInt(Integer::intValue).toArray();
        
        if (selectedFeatureIndices.length == 0) {
            throw new IllegalStateException(
                "No features meet the variance threshold " + threshold + 
                ". Consider lowering the threshold.");
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Reduces X to the selected features.
     * 
     * @param X Data to transform of shape [n_samples, n_features]
     * @return Transformed data with only selected features [n_samples, n_selected_features]
     * @throws IllegalStateException if fit() has not been called
     * @throws IllegalArgumentException if X has different number of features than training data
     */
    public double[][] transform(double[][] X) {
        checkFitted();
        validateInput(X);
        
        if (X[0].length != variances.length) {
            throw new IllegalArgumentException(
                "X has " + X[0].length + " features, but VarianceThreshold was fitted with " + 
                variances.length + " features");
        }
        
        int nSamples = X.length;
        int nSelectedFeatures = selectedFeatureIndices.length;
        double[][] result = new double[nSamples][nSelectedFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nSelectedFeatures; j++) {
                result[i][j] = X[i][selectedFeatureIndices[j]];
            }
        }
        
        return result;
    }
    
    /**
     * Fits the selector and transforms the data in one step.
     * 
     * @param X Training data of shape [n_samples, n_features]
     * @return Transformed data with only selected features
     */
    public double[][] fitTransform(double[][] X) {
        return fit(X).transform(X);
    }
    
    /**
     * Gets the variance of each feature computed during fit.
     * 
     * @return Array of variances for each feature
     * @throws IllegalStateException if fit() has not been called
     */
    public double[] getVariances() {
        checkFitted();
        return variances.clone();
    }
    
    /**
     * Gets the indices of selected features.
     * 
     * @return Array of indices of features that passed the variance threshold
     * @throws IllegalStateException if fit() has not been called
     */
    public int[] getSelectedFeatureIndices() {
        checkFitted();
        return selectedFeatureIndices.clone();
    }
    
    /**
     * Gets a boolean mask of selected features.
     * 
     * @return Boolean array where true indicates the feature is selected
     * @throws IllegalStateException if fit() has not been called
     */
    public boolean[] getSupport() {
        checkFitted();
        boolean[] support = new boolean[variances.length];
        for (int idx : selectedFeatureIndices) {
            support[idx] = true;
        }
        return support;
    }
    
    /**
     * Gets the number of selected features.
     * 
     * @return Number of features after selection
     * @throws IllegalStateException if fit() has not been called
     */
    public int getNumberOfSelectedFeatures() {
        checkFitted();
        return selectedFeatureIndices.length;
    }
    
    /**
     * Gets the threshold value.
     * 
     * @return The variance threshold
     */
    public double getThreshold() {
        return threshold;
    }
    
    /**
     * Checks if the selector has been fitted.
     * 
     * @return true if fit() has been called
     */
    public boolean isFitted() {
        return fitted;
    }
    
    private void validateInput(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data X cannot be null or empty");
        }
        if (X[0] == null || X[0].length == 0) {
            throw new IllegalArgumentException("Input data X must have at least one feature");
        }
        
        // Check all rows have same number of features
        int nFeatures = X[0].length;
        for (int i = 1; i < X.length; i++) {
            if (X[i] == null || X[i].length != nFeatures) {
                throw new IllegalArgumentException(
                    "All samples must have the same number of features");
            }
        }
    }
    
    private void checkFitted() {
        if (!fitted) {
            throw new IllegalStateException(
                "This VarianceThreshold instance is not fitted yet. " +
                "Call 'fit' with appropriate arguments before using this method.");
        }
    }
    
    @Override
    public String toString() {
        if (fitted) {
            return String.format("VarianceThreshold(threshold=%.4f, n_features_in=%d, n_features_out=%d)",
                threshold, variances.length, selectedFeatureIndices.length);
        } else {
            return String.format("VarianceThreshold(threshold=%.4f, fitted=false)", threshold);
        }
    }
}
