package com.mindforge.preprocessing;

import java.util.Arrays;

/**
 * MinMaxScaler transforms features by scaling each feature to a given range (default: [0, 1]).
 * 
 * Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
 * 
 * Example:
 * <pre>
 * double[][] data = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
 * MinMaxScaler scaler = new MinMaxScaler();
 * double[][] scaled = scaler.fitTransform(data);
 * </pre>
 */
public class MinMaxScaler {
    private double[] featureMin;
    private double[] featureMax;
    private double minRange;
    private double maxRange;
    private boolean fitted;

    /**
     * Creates a MinMaxScaler with default range [0, 1].
     */
    public MinMaxScaler() {
        this(0.0, 1.0);
    }

    /**
     * Creates a MinMaxScaler with custom range.
     * 
     * @param minRange minimum value of the desired range
     * @param maxRange maximum value of the desired range
     */
    public MinMaxScaler(double minRange, double maxRange) {
        if (minRange >= maxRange) {
            throw new IllegalArgumentException("minRange must be less than maxRange");
        }
        this.minRange = minRange;
        this.maxRange = maxRange;
        this.fitted = false;
    }

    /**
     * Computes the minimum and maximum values for each feature.
     * 
     * @param X training data of shape (n_samples, n_features)
     */
    public void fit(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }

        int nSamples = X.length;
        int nFeatures = X[0].length;

        featureMin = new double[nFeatures];
        featureMax = new double[nFeatures];
        Arrays.fill(featureMin, Double.POSITIVE_INFINITY);
        Arrays.fill(featureMax, Double.NEGATIVE_INFINITY);

        // Find min and max for each feature
        for (int i = 0; i < nSamples; i++) {
            if (X[i].length != nFeatures) {
                throw new IllegalArgumentException("All rows must have the same number of features");
            }
            for (int j = 0; j < nFeatures; j++) {
                featureMin[j] = Math.min(featureMin[j], X[i][j]);
                featureMax[j] = Math.max(featureMax[j], X[i][j]);
            }
        }

        this.fitted = true;
    }

    /**
     * Scales features according to the fitted min and max values.
     * 
     * @param X data to transform
     * @return scaled data
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

        if (nFeatures != featureMin.length) {
            throw new IllegalArgumentException(
                "Input has " + nFeatures + " features, but scaler was fitted with " + featureMin.length + " features"
            );
        }

        double[][] scaled = new double[nSamples][nFeatures];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                double range = featureMax[j] - featureMin[j];
                if (range == 0) {
                    // If all values are the same, scale to the middle of the range
                    scaled[i][j] = (minRange + maxRange) / 2.0;
                } else {
                    scaled[i][j] = ((X[i][j] - featureMin[j]) / range) * (maxRange - minRange) + minRange;
                }
            }
        }

        return scaled;
    }

    /**
     * Fits the scaler and transforms the data in one step.
     * 
     * @param X data to fit and transform
     * @return scaled data
     */
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }

    /**
     * Reverses the scaling transformation.
     * 
     * @param X scaled data
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
                double range = featureMax[j] - featureMin[j];
                if (range == 0) {
                    original[i][j] = featureMin[j];
                } else {
                    original[i][j] = ((X[i][j] - minRange) / (maxRange - minRange)) * range + featureMin[j];
                }
            }
        }

        return original;
    }

    /**
     * Gets the minimum values for each feature.
     * 
     * @return array of minimum values
     */
    public double[] getFeatureMin() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted yet");
        }
        return featureMin.clone();
    }

    /**
     * Gets the maximum values for each feature.
     * 
     * @return array of maximum values
     */
    public double[] getFeatureMax() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted yet");
        }
        return featureMax.clone();
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
