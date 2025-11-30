package com.mindforge.preprocessing;

import java.util.Arrays;

/**
 * SimpleImputer for completing missing values in datasets.
 * 
 * Supports different imputation strategies:
 * - MEAN: Replace missing values with the mean of the column
 * - MEDIAN: Replace missing values with the median of the column
 * - MOST_FREQUENT: Replace missing values with the most frequent value
 * - CONSTANT: Replace missing values with a constant value
 * 
 * Missing values are represented as Double.NaN.
 * 
 * Example:
 * <pre>
 * double[][] data = {{1.0, 2.0}, {Double.NaN, 3.0}, {7.0, Double.NaN}};
 * SimpleImputer imputer = new SimpleImputer(ImputeStrategy.MEAN);
 * double[][] filled = imputer.fitTransform(data);
 * </pre>
 */
public class SimpleImputer {
    
    public enum ImputeStrategy {
        MEAN,
        MEDIAN,
        MOST_FREQUENT,
        CONSTANT
    }

    private ImputeStrategy strategy;
    private double fillValue;
    private double[] statistics;
    private boolean fitted;

    /**
     * Creates an imputer with the specified strategy.
     * 
     * @param strategy the imputation strategy to use
     */
    public SimpleImputer(ImputeStrategy strategy) {
        this(strategy, 0.0);
    }

    /**
     * Creates an imputer with CONSTANT strategy.
     * 
     * @param strategy the imputation strategy (should be CONSTANT)
     * @param fillValue the constant value to use for imputation
     */
    public SimpleImputer(ImputeStrategy strategy, double fillValue) {
        this.strategy = strategy;
        this.fillValue = fillValue;
        this.fitted = false;
    }

    /**
     * Fits the imputer on the data by computing the statistics.
     * 
     * @param X data with missing values
     */
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }

        int nFeatures = X[0].length;
        statistics = new double[nFeatures];

        if (strategy == ImputeStrategy.CONSTANT) {
            Arrays.fill(statistics, fillValue);
        } else {
            for (int j = 0; j < nFeatures; j++) {
                statistics[j] = computeStatistic(X, j);
            }
        }

        this.fitted = true;
    }

    /**
     * Transforms the data by replacing missing values.
     * 
     * @param X data with missing values
     * @return data with imputed values
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Imputer must be fitted before transformation");
        }
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }

        int nSamples = X.length;
        int nFeatures = X[0].length;

        if (nFeatures != statistics.length) {
            throw new IllegalArgumentException(
                "Input has " + nFeatures + " features, but imputer was fitted with " + statistics.length + " features"
            );
        }

        double[][] result = new double[nSamples][nFeatures];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                if (Double.isNaN(X[i][j])) {
                    result[i][j] = statistics[j];
                } else {
                    result[i][j] = X[i][j];
                }
            }
        }

        return result;
    }

    /**
     * Fits the imputer and transforms the data in one step.
     * 
     * @param X data with missing values
     * @return data with imputed values
     */
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }

    /**
     * Computes the statistic for a given column based on the strategy.
     * 
     * @param X data matrix
     * @param column column index
     * @return the computed statistic
     */
    private double computeStatistic(double[][] X, int column) {
        switch (strategy) {
            case MEAN:
                return computeMean(X, column);
            case MEDIAN:
                return computeMedian(X, column);
            case MOST_FREQUENT:
                return computeMostFrequent(X, column);
            default:
                return fillValue;
        }
    }

    /**
     * Computes the mean of a column, ignoring NaN values.
     */
    private double computeMean(double[][] X, int column) {
        double sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < X.length; i++) {
            if (!Double.isNaN(X[i][column])) {
                sum += X[i][column];
                count++;
            }
        }
        
        if (count == 0) {
            return 0.0; // All values are NaN
        }
        
        return sum / count;
    }

    /**
     * Computes the median of a column, ignoring NaN values.
     */
    private double computeMedian(double[][] X, int column) {
        double[] values = new double[X.length];
        int count = 0;
        
        for (int i = 0; i < X.length; i++) {
            if (!Double.isNaN(X[i][column])) {
                values[count++] = X[i][column];
            }
        }
        
        if (count == 0) {
            return 0.0; // All values are NaN
        }
        
        double[] validValues = Arrays.copyOf(values, count);
        Arrays.sort(validValues);
        
        if (count % 2 == 0) {
            return (validValues[count / 2 - 1] + validValues[count / 2]) / 2.0;
        } else {
            return validValues[count / 2];
        }
    }

    /**
     * Computes the most frequent value in a column, ignoring NaN values.
     */
    private double computeMostFrequent(double[][] X, int column) {
        java.util.Map<Double, Integer> frequency = new java.util.HashMap<>();
        
        for (int i = 0; i < X.length; i++) {
            if (!Double.isNaN(X[i][column])) {
                double value = X[i][column];
                frequency.put(value, frequency.getOrDefault(value, 0) + 1);
            }
        }
        
        if (frequency.isEmpty()) {
            return 0.0; // All values are NaN
        }
        
        return frequency.entrySet().stream()
                .max(java.util.Map.Entry.comparingByValue())
                .get()
                .getKey();
    }

    /**
     * Gets the computed statistics for each feature.
     * 
     * @return array of statistics
     */
    public double[] getStatistics() {
        if (!fitted) {
            throw new IllegalStateException("Imputer has not been fitted yet");
        }
        return statistics.clone();
    }

    /**
     * Checks if the imputer has been fitted.
     * 
     * @return true if fitted, false otherwise
     */
    public boolean isFitted() {
        return fitted;
    }
}
