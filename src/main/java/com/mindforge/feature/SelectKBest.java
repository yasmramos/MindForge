package com.mindforge.feature;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Select features according to the k highest scores.
 * 
 * This feature selector uses statistical tests to select features
 * with the strongest relationship to the target variable.
 * 
 * Available scoring functions:
 * - F_CLASSIF: ANOVA F-value for classification
 * - CHI2: Chi-squared stats for non-negative features
 * - MUTUAL_INFO: Mutual information for classification
 * 
 * Example usage:
 * <pre>
 * SelectKBest selector = new SelectKBest(SelectKBest.ScoreFunction.F_CLASSIF, 5);
 * selector.fit(X, y);
 * double[][] X_selected = selector.transform(X);
 * // Or in one step:
 * double[][] X_selected = selector.fitTransform(X, y);
 * </pre>
 * 
 * @author MindForge
 * @version 1.0.8-alpha
 */
public class SelectKBest {
    
    /**
     * Available scoring functions for feature selection.
     */
    public enum ScoreFunction {
        /** ANOVA F-value between feature and target */
        F_CLASSIF,
        /** Chi-squared stats (requires non-negative features) */
        CHI2,
        /** Mutual information for discrete target */
        MUTUAL_INFO
    }
    
    private final ScoreFunction scoreFunction;
    private final int k;
    private double[] scores;
    private double[] pValues;
    private int[] selectedFeatureIndices;
    private int nFeaturesIn;
    private boolean fitted;
    
    /**
     * Creates a SelectKBest selector with F_CLASSIF scoring.
     * 
     * @param k Number of top features to select
     */
    public SelectKBest(int k) {
        this(ScoreFunction.F_CLASSIF, k);
    }
    
    /**
     * Creates a SelectKBest selector with specified scoring function.
     * 
     * @param scoreFunction The scoring function to use
     * @param k Number of top features to select. Use -1 for "all" (keeps all features)
     * @throws IllegalArgumentException if k is less than 1 and not -1
     */
    public SelectKBest(ScoreFunction scoreFunction, int k) {
        if (k < 1 && k != -1) {
            throw new IllegalArgumentException("k must be positive or -1 for 'all', got: " + k);
        }
        this.scoreFunction = scoreFunction;
        this.k = k;
        this.fitted = false;
    }
    
    /**
     * Computes feature scores and identifies the k best features.
     * 
     * @param X Training data of shape [n_samples, n_features]
     * @param y Target values of shape [n_samples]
     * @return this selector for method chaining
     * @throws IllegalArgumentException if inputs are invalid
     */
    public SelectKBest fit(double[][] X, int[] y) {
        validateInput(X, y);
        
        nFeaturesIn = X[0].length;
        int actualK = (k == -1) ? nFeaturesIn : Math.min(k, nFeaturesIn);
        
        // Calculate scores based on selected function
        switch (scoreFunction) {
            case F_CLASSIF:
                calculateFClassif(X, y);
                break;
            case CHI2:
                calculateChi2(X, y);
                break;
            case MUTUAL_INFO:
                calculateMutualInfo(X, y);
                break;
        }
        
        // Select top k features
        Integer[] indices = new Integer[nFeaturesIn];
        for (int i = 0; i < nFeaturesIn; i++) {
            indices[i] = i;
        }
        
        // Sort by score descending
        Arrays.sort(indices, (a, b) -> Double.compare(scores[b], scores[a]));
        
        // Take top k
        selectedFeatureIndices = new int[actualK];
        for (int i = 0; i < actualK; i++) {
            selectedFeatureIndices[i] = indices[i];
        }
        
        // Sort selected indices for consistent ordering
        Arrays.sort(selectedFeatureIndices);
        
        fitted = true;
        return this;
    }
    
    /**
     * Calculates ANOVA F-value for each feature.
     * F-value measures the linear dependency between the feature and target.
     */
    private void calculateFClassif(double[][] X, int[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Find unique classes
        int[] uniqueClasses = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = uniqueClasses.length;
        
        scores = new double[nFeatures];
        pValues = new double[nFeatures];
        
        for (int j = 0; j < nFeatures; j++) {
            // Calculate overall mean
            double overallMean = 0.0;
            for (int i = 0; i < nSamples; i++) {
                overallMean += X[i][j];
            }
            overallMean /= nSamples;
            
            // Calculate between-group and within-group variance
            double ssBetween = 0.0;
            double ssWithin = 0.0;
            
            for (int c : uniqueClasses) {
                // Get samples for this class
                double classMean = 0.0;
                int classCount = 0;
                
                for (int i = 0; i < nSamples; i++) {
                    if (y[i] == c) {
                        classMean += X[i][j];
                        classCount++;
                    }
                }
                classMean /= classCount;
                
                // Between-group sum of squares
                ssBetween += classCount * Math.pow(classMean - overallMean, 2);
                
                // Within-group sum of squares
                for (int i = 0; i < nSamples; i++) {
                    if (y[i] == c) {
                        ssWithin += Math.pow(X[i][j] - classMean, 2);
                    }
                }
            }
            
            // Calculate F-value
            double dfBetween = nClasses - 1;
            double dfWithin = nSamples - nClasses;
            
            double msBetween = ssBetween / dfBetween;
            double msWithin = ssWithin / dfWithin;
            
            if (msWithin > 0) {
                scores[j] = msBetween / msWithin;
            } else {
                scores[j] = 0.0;
            }
            
            // P-value approximation (using F-distribution would require more complex code)
            pValues[j] = approximateFPValue(scores[j], (int) dfBetween, (int) dfWithin);
        }
    }
    
    /**
     * Calculates Chi-squared statistics for each feature.
     * Requires non-negative feature values.
     */
    private void calculateChi2(double[][] X, int[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        // Verify non-negative features
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                if (X[i][j] < 0) {
                    throw new IllegalArgumentException(
                        "Chi2 requires non-negative feature values. Found negative value at [" + 
                        i + "][" + j + "]");
                }
            }
        }
        
        int[] uniqueClasses = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = uniqueClasses.length;
        
        scores = new double[nFeatures];
        pValues = new double[nFeatures];
        
        // Calculate feature sums per class and total
        double[][] featureSumPerClass = new double[nClasses][nFeatures];
        double[] featureSumTotal = new double[nFeatures];
        int[] classCounts = new int[nClasses];
        
        for (int i = 0; i < nSamples; i++) {
            int classIdx = Arrays.binarySearch(uniqueClasses, y[i]);
            classCounts[classIdx]++;
            for (int j = 0; j < nFeatures; j++) {
                featureSumPerClass[classIdx][j] += X[i][j];
                featureSumTotal[j] += X[i][j];
            }
        }
        
        // Calculate chi-squared for each feature
        for (int j = 0; j < nFeatures; j++) {
            double chi2 = 0.0;
            
            for (int c = 0; c < nClasses; c++) {
                // Expected value
                double expected = (featureSumTotal[j] * classCounts[c]) / nSamples;
                
                if (expected > 0) {
                    double observed = featureSumPerClass[c][j];
                    chi2 += Math.pow(observed - expected, 2) / expected;
                }
            }
            
            scores[j] = chi2;
            pValues[j] = approximateChi2PValue(chi2, nClasses - 1);
        }
    }
    
    /**
     * Calculates mutual information between each feature and the target.
     */
    private void calculateMutualInfo(double[][] X, int[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        
        int[] uniqueClasses = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = uniqueClasses.length;
        
        scores = new double[nFeatures];
        pValues = new double[nFeatures]; // MI doesn't have p-values, set to 0
        
        // Class probabilities
        double[] classProbs = new double[nClasses];
        for (int label : y) {
            int classIdx = Arrays.binarySearch(uniqueClasses, label);
            classProbs[classIdx]++;
        }
        for (int c = 0; c < nClasses; c++) {
            classProbs[c] /= nSamples;
        }
        
        // Calculate MI for each feature using discretization
        int nBins = Math.min(10, (int) Math.sqrt(nSamples));
        
        for (int j = 0; j < nFeatures; j++) {
            // Find min and max for binning
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (int i = 0; i < nSamples; i++) {
                min = Math.min(min, X[i][j]);
                max = Math.max(max, X[i][j]);
            }
            
            double binWidth = (max - min) / nBins;
            if (binWidth == 0) binWidth = 1.0;
            
            // Count joint and marginal frequencies
            int[][] jointCounts = new int[nBins][nClasses];
            int[] binCounts = new int[nBins];
            
            for (int i = 0; i < nSamples; i++) {
                int bin = Math.min((int) ((X[i][j] - min) / binWidth), nBins - 1);
                int classIdx = Arrays.binarySearch(uniqueClasses, y[i]);
                jointCounts[bin][classIdx]++;
                binCounts[bin]++;
            }
            
            // Calculate mutual information
            double mi = 0.0;
            for (int b = 0; b < nBins; b++) {
                double pBin = (double) binCounts[b] / nSamples;
                if (pBin > 0) {
                    for (int c = 0; c < nClasses; c++) {
                        double pJoint = (double) jointCounts[b][c] / nSamples;
                        if (pJoint > 0) {
                            mi += pJoint * Math.log(pJoint / (pBin * classProbs[c]));
                        }
                    }
                }
            }
            
            scores[j] = Math.max(0, mi); // MI should be non-negative
        }
    }
    
    /**
     * Approximate p-value for F-distribution (simplified).
     */
    private double approximateFPValue(double f, int df1, int df2) {
        if (f <= 0) return 1.0;
        // Very rough approximation
        double x = df2 / (df2 + df1 * f);
        return Math.max(0, Math.min(1, x));
    }
    
    /**
     * Approximate p-value for Chi-squared distribution (simplified).
     */
    private double approximateChi2PValue(double chi2, int df) {
        if (chi2 <= 0) return 1.0;
        // Very rough approximation using normal approximation for large df
        double z = Math.pow(chi2 / df, 1.0/3.0) - (1.0 - 2.0/(9.0*df));
        z /= Math.sqrt(2.0/(9.0*df));
        return Math.max(0, Math.min(1, 0.5 * (1 - erf(z / Math.sqrt(2)))));
    }
    
    /**
     * Error function approximation.
     */
    private double erf(double x) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double tau = t * Math.exp(-x*x - 1.26551223 +
            t * (1.00002368 +
            t * (0.37409196 +
            t * (0.09678418 +
            t * (-0.18628806 +
            t * (0.27886807 +
            t * (-1.13520398 +
            t * (1.48851587 +
            t * (-0.82215223 +
            t * 0.17087277)))))))));
        return x >= 0 ? 1 - tau : tau - 1;
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
                "X has " + X[0].length + " features, but SelectKBest was fitted with " + 
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
     * Gets the scores for all features.
     * 
     * @return Array of scores for each feature
     */
    public double[] getScores() {
        checkFitted();
        return scores.clone();
    }
    
    /**
     * Gets the p-values for all features (if applicable).
     * 
     * @return Array of p-values for each feature
     */
    public double[] getPValues() {
        checkFitted();
        return pValues.clone();
    }
    
    /**
     * Gets the indices of selected features.
     * 
     * @return Array of indices of selected features
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
     * @return The k value
     */
    public int getK() {
        return k;
    }
    
    /**
     * Gets the scoring function used.
     * 
     * @return The score function
     */
    public ScoreFunction getScoreFunction() {
        return scoreFunction;
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
                "This SelectKBest instance is not fitted yet. " +
                "Call 'fit' with appropriate arguments before using this method.");
        }
    }
    
    @Override
    public String toString() {
        if (fitted) {
            return String.format("SelectKBest(scoreFunction=%s, k=%d, n_features_in=%d, n_features_out=%d)",
                scoreFunction, k, nFeaturesIn, selectedFeatureIndices.length);
        } else {
            return String.format("SelectKBest(scoreFunction=%s, k=%d, fitted=false)", scoreFunction, k);
        }
    }
}
