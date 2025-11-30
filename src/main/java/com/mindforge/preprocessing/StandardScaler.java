package com.mindforge.preprocessing;

import java.util.Arrays;

/**
 * StandardScaler standardizes features by removing the mean and scaling to unit variance.
 * 
 * Formula: X_scaled = (X - mean) / std
 * 
 * This is also known as Z-score normalization.
 * 
 * Example:
 * <pre>
 * double[][] data = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
 * StandardScaler scaler = new StandardScaler();
 * double[][] scaled = scaler.fitTransform(data);
 * </pre>
 */
public class StandardScaler {
    private double[] mean;
    private double[] std;
    private boolean withMean;
    private boolean withStd;
    private boolean fitted;

    /**
     * Creates a StandardScaler with default settings (center and scale).
     */
    public StandardScaler() {
        this(true, true);
    }

    /**
     * Creates a StandardScaler with custom settings.
     * 
     * @param withMean if true, center the data before scaling
     * @param withStd if true, scale the data to unit variance
     */
    public StandardScaler(boolean withMean, boolean withStd) {
        this.withMean = withMean;
        this.withStd = withStd;
        this.fitted = false;
    }

    /**
     * Computes the mean and standard deviation for each feature.
     * 
     * @param X training data of shape (n_samples, n_features)
     */
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }

        int nSamples = X.length;
        int nFeatures = X[0].length;

        mean = new double[nFeatures];
        std = new double[nFeatures];

        // Compute mean
        for (int i = 0; i < nSamples; i++) {
            if (X[i].length != nFeatures) {
                throw new IllegalArgumentException("All rows must have the same number of features");
            }
            for (int j = 0; j < nFeatures; j++) {
                mean[j] += X[i][j];
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            mean[j] /= nSamples;
        }

        // Compute standard deviation
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double diff = X[i][j] - mean[j];
                std[j] += diff * diff;
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            std[j] = Math.sqrt(std[j] / nSamples);
        }

        this.fitted = true;
    }

    /**
     * Standardizes features according to the fitted mean and standard deviation.
     * 
     * @param X data to transform
     * @return standardized data
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transformation");
        }
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }

        int nSamples = X.length;
        int nFeatures = X[0].length;

        if (nFeatures != mean.length) {
            throw new IllegalArgumentException(
                "Input has " + nFeatures + " features, but scaler was fitted with " + mean.length + " features"
            );
        }

        double[][] scaled = new double[nSamples][nFeatures];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = X[i][j];
                
                if (withMean) {
                    value -= mean[j];
                }
                
                if (withStd) {
                    if (std[j] > 0) {
                        value /= std[j];
                    }
                    // If std is 0, the feature is constant, keep it as is
                }
                
                scaled[i][j] = value;
            }
        }

        return scaled;
    }

    /**
     * Fits the scaler and transforms the data in one step.
     * 
     * @param X data to fit and transform
     * @return standardized data
     */
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }

    /**
     * Reverses the standardization transformation.
     * 
     * @param X standardized data
     * @return original scale data
     */
    public double[][] inverseTransform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before inverse transformation");
        }
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }

        int nSamples = X.length;
        int nFeatures = X[0].length;

        double[][] original = new double[nSamples][nFeatures];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double value = X[i][j];
                
                if (withStd && std[j] > 0) {
                    value *= std[j];
                }
                
                if (withMean) {
                    value += mean[j];
                }
                
                original[i][j] = value;
            }
        }

        return original;
    }

    /**
     * Gets the mean values for each feature.
     * 
     * @return array of mean values
     */
    public double[] getMean() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted yet");
        }
        return mean.clone();
    }

    /**
     * Gets the standard deviation values for each feature.
     * 
     * @return array of standard deviation values
     */
    public double[] getStd() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted yet");
        }
        return std.clone();
    }

    /**
     * Checks if the scaler has been fitted.
     * 
     * @return true if fitted, false otherwise
     */
    public boolean isFitted() {
        return fitted;
    }
}
