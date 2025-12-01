package com.mindforge.feature;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Recursive Feature Elimination (RFE) for feature selection.
 * 
 * RFE works by recursively removing features and building a model on
 * the remaining features. It uses feature importance or coefficients
 * from the model to identify which features to eliminate.
 * 
 * This implementation uses a simple linear model internally to compute
 * feature importance based on correlation with the target.
 * 
 * Example usage:
 * <pre>
 * RFE rfe = new RFE(5); // Select 5 best features
 * rfe.fit(X, y);
 * double[][] X_selected = rfe.transform(X);
 * // Or with step parameter:
 * RFE rfe = new RFE(5, 2); // Remove 2 features per iteration
 * </pre>
 * 
 * @author MindForge
 * @version 1.0.8-alpha
 */
public class RFE {
    
    private final int nFeaturesToSelect;
    private final int step;
    private int[] ranking;
    private int[] selectedFeatureIndices;
    private double[] featureImportances;
    private int nFeaturesIn;
    private boolean fitted;
    
    /**
     * Creates an RFE selector with default step of 1.
     * 
     * @param nFeaturesToSelect Number of features to select
     */
    public RFE(int nFeaturesToSelect) {
        this(nFeaturesToSelect, 1);
    }
    
    /**
     * Creates an RFE selector with specified parameters.
     * 
     * @param nFeaturesToSelect Number of features to select. If -1, selects half of the features.
     * @param step Number of features to remove at each iteration. Must be >= 1.
     * @throws IllegalArgumentException if parameters are invalid
     */
    public RFE(int nFeaturesToSelect, int step) {
        if (nFeaturesToSelect < 1 && nFeaturesToSelect != -1) {
            throw new IllegalArgumentException(
                "nFeaturesToSelect must be positive or -1, got: " + nFeaturesToSelect);
        }
        if (step < 1) {
            throw new IllegalArgumentException("step must be at least 1, got: " + step);
        }
        this.nFeaturesToSelect = nFeaturesToSelect;
        this.step = step;
        this.fitted = false;
    }
    
    /**
     * Fits the RFE selector using recursive feature elimination.
     * 
     * @param X Training data of shape [n_samples, n_features]
     * @param y Target values of shape [n_samples]
     * @return this selector for method chaining
     */
    public RFE fit(double[][] X, int[] y) {
        validateInput(X, y);
        
        nFeaturesIn = X[0].length;
        int actualNFeatures = (nFeaturesToSelect == -1) ? nFeaturesIn / 2 : 
            Math.min(nFeaturesToSelect, nFeaturesIn);
        
        if (actualNFeatures < 1) {
            actualNFeatures = 1;
        }
        
        // Convert y to double for correlation calculation
        double[] yDouble = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            yDouble[i] = y[i];
        }
        
        // Initialize: all features are candidates
        List<Integer> remainingFeatures = new ArrayList<>();
        for (int i = 0; i < nFeaturesIn; i++) {
            remainingFeatures.add(i);
        }
        
        // Track ranking (1 = best/selected, higher = eliminated earlier)
        ranking = new int[nFeaturesIn];
        Arrays.fill(ranking, 1);
        
        // Store all feature importances (from the last full calculation)
        featureImportances = new double[nFeaturesIn];
        
        int currentRank = nFeaturesIn;
        
        // Recursively eliminate features
        while (remainingFeatures.size() > actualNFeatures) {
            // Calculate importance for remaining features
            double[] importances = calculateFeatureImportances(X, yDouble, remainingFeatures);
            
            // Store importances for remaining features
            for (int i = 0; i < remainingFeatures.size(); i++) {
                featureImportances[remainingFeatures.get(i)] = importances[i];
            }
            
            // Find features to eliminate (those with lowest importance)
            int nToEliminate = Math.min(step, remainingFeatures.size() - actualNFeatures);
            
            // Get indices sorted by importance (ascending)
            Integer[] sortedIndices = new Integer[importances.length];
            for (int i = 0; i < sortedIndices.length; i++) {
                sortedIndices[i] = i;
            }
            final double[] impFinal = importances;
            Arrays.sort(sortedIndices, (a, b) -> Double.compare(impFinal[a], impFinal[b]));
            
            // Eliminate the least important features
            List<Integer> toRemove = new ArrayList<>();
            for (int i = 0; i < nToEliminate; i++) {
                int featureIdx = remainingFeatures.get(sortedIndices[i]);
                ranking[featureIdx] = currentRank--;
                toRemove.add(remainingFeatures.get(sortedIndices[i]));
            }
            
            remainingFeatures.removeAll(toRemove);
        }
        
        // Calculate final importances for selected features
        double[] finalImportances = calculateFeatureImportances(X, yDouble, remainingFeatures);
        for (int i = 0; i < remainingFeatures.size(); i++) {
            featureImportances[remainingFeatures.get(i)] = finalImportances[i];
        }
        
        // Store selected features (sorted by index)
        selectedFeatureIndices = remainingFeatures.stream()
            .mapToInt(Integer::intValue)
            .sorted()
            .toArray();
        
        fitted = true;
        return this;
    }
    
    /**
     * Calculates feature importances using absolute correlation with target.
     */
    private double[] calculateFeatureImportances(double[][] X, double[] y, 
            List<Integer> featureIndices) {
        int nSamples = X.length;
        int nFeatures = featureIndices.size();
        double[] importances = new double[nFeatures];
        
        // Calculate mean of y
        double yMean = 0.0;
        for (double val : y) {
            yMean += val;
        }
        yMean /= nSamples;
        
        // Calculate variance of y
        double yVar = 0.0;
        for (double val : y) {
            yVar += Math.pow(val - yMean, 2);
        }
        
        // Calculate correlation coefficient for each feature
        for (int j = 0; j < nFeatures; j++) {
            int featureIdx = featureIndices.get(j);
            
            // Calculate mean of feature
            double xMean = 0.0;
            for (int i = 0; i < nSamples; i++) {
                xMean += X[i][featureIdx];
            }
            xMean /= nSamples;
            
            // Calculate variance of feature and covariance with y
            double xVar = 0.0;
            double covar = 0.0;
            for (int i = 0; i < nSamples; i++) {
                double xDiff = X[i][featureIdx] - xMean;
                double yDiff = y[i] - yMean;
                xVar += xDiff * xDiff;
                covar += xDiff * yDiff;
            }
            
            // Correlation coefficient (Pearson)
            double denom = Math.sqrt(xVar * yVar);
            if (denom > 0) {
                importances[j] = Math.abs(covar / denom);
            } else {
                importances[j] = 0.0;
            }
        }
        
        return importances;
    }
    
    /**
     * Reduces X to the selected features.
     * 
     * @param X Data to transform of shape [n_samples, n_features]
     * @return Transformed data with only selected features
     */
    public double[][] transform(double[][] X) {
        checkFitted();
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data X cannot be null or empty");
        }
        if (X[0].length != nFeaturesIn) {
            throw new IllegalArgumentException(
                "X has " + X[0].length + " features, but RFE was fitted with " + 
                nFeaturesIn + " features");
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
     * @param X Training data
     * @param y Target values
     * @return Transformed data with only selected features
     */
    public double[][] fitTransform(double[][] X, int[] y) {
        return fit(X, y).transform(X);
    }
    
    /**
     * Gets the ranking of each feature.
     * Selected features have ranking 1, and eliminated features
     * have higher rankings based on when they were eliminated.
     * 
     * @return Array of rankings for each feature
     */
    public int[] getRanking() {
        checkFitted();
        return ranking.clone();
    }
    
    /**
     * Gets the feature importances from the final model.
     * 
     * @return Array of importance scores for each feature
     */
    public double[] getFeatureImportances() {
        checkFitted();
        return featureImportances.clone();
    }
    
    /**
     * Gets the indices of selected features.
     * 
     * @return Array of indices of selected features (sorted)
     */
    public int[] getSelectedFeatureIndices() {
        checkFitted();
        return selectedFeatureIndices.clone();
    }
    
    /**
     * Gets a boolean mask of selected features.
     * 
     * @return Boolean array where true indicates the feature is selected
     */
    public boolean[] getSupport() {
        checkFitted();
        boolean[] support = new boolean[nFeaturesIn];
        for (int idx : selectedFeatureIndices) {
            support[idx] = true;
        }
        return support;
    }
    
    /**
     * Gets the number of features to select.
     * 
     * @return The target number of features
     */
    public int getNFeaturesToSelect() {
        return nFeaturesToSelect;
    }
    
    /**
     * Gets the step size used during elimination.
     * 
     * @return The step value
     */
    public int getStep() {
        return step;
    }
    
    /**
     * Gets the number of selected features.
     * 
     * @return Number of features after selection
     */
    public int getNumberOfSelectedFeatures() {
        checkFitted();
        return selectedFeatureIndices.length;
    }
    
    /**
     * Checks if the selector has been fitted.
     * 
     * @return true if fit() has been called
     */
    public boolean isFitted() {
        return fitted;
    }
    
    private void validateInput(double[][] X, int[] y) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data X cannot be null or empty");
        }
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Target y cannot be null or empty");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                "X and y must have the same number of samples. X has " + 
                X.length + ", y has " + y.length);
        }
        if (X[0] == null || X[0].length == 0) {
            throw new IllegalArgumentException("Input data X must have at least one feature");
        }
    }
    
    private void checkFitted() {
        if (!fitted) {
            throw new IllegalStateException(
                "This RFE instance is not fitted yet. " +
                "Call 'fit' with appropriate arguments before using this method.");
        }
    }
    
    @Override
    public String toString() {
        if (fitted) {
            return String.format("RFE(nFeaturesToSelect=%d, step=%d, n_features_in=%d, n_features_out=%d)",
                nFeaturesToSelect, step, nFeaturesIn, selectedFeatureIndices.length);
        } else {
            return String.format("RFE(nFeaturesToSelect=%d, step=%d, fitted=false)", 
                nFeaturesToSelect, step);
        }
    }
}
