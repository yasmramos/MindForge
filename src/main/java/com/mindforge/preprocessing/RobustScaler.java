package com.mindforge.preprocessing;

import java.util.Arrays;

/**
 * RobustScaler scales features using statistics that are robust to outliers.
 * 
 * This scaler removes the median and scales the data according to the
 * Interquartile Range (IQR). The IQR is the range between the 1st quartile
 * (25th percentile) and the 3rd quartile (75th percentile).
 * 
 * Formula: X_scaled = (X - median) / IQR
 * 
 * Unlike StandardScaler which uses mean and standard deviation, RobustScaler
 * uses statistics that are robust to outliers:
 * - Median instead of mean
 * - Interquartile Range (IQR) instead of standard deviation
 * 
 * This makes RobustScaler particularly useful when the data contains outliers.
 * 
 * Key features:
 * - Configurable centering (median subtraction)
 * - Configurable scaling (IQR division)
 * - Custom quantile range support
 * - Unit variance option
 * - Inverse transform support
 * 
 * Example usage:
 * <pre>
 * // Default usage (center and scale)
 * RobustScaler scaler = new RobustScaler.Builder().build();
 * double[][] scaled = scaler.fitTransform(data);
 * 
 * // Custom quantile range (e.g., 5th to 95th percentile)
 * RobustScaler scaler = new RobustScaler.Builder()
 *     .quantileRange(5.0, 95.0)
 *     .build();
 * 
 * // Scale without centering
 * RobustScaler scaler = new RobustScaler.Builder()
 *     .withCentering(false)
 *     .build();
 * </pre>
 * 
 * @author MindForge Team
 * @since 2.0.0
 */
public class RobustScaler {
    
    // Configuration
    private final boolean withCentering;
    private final boolean withScaling;
    private final double quantileMin;  // Lower quantile (e.g., 25.0)
    private final double quantileMax;  // Upper quantile (e.g., 75.0)
    private final boolean unitVariance;
    
    // Learned parameters
    private double[] center;           // Median for each feature
    private double[] scale;            // IQR for each feature
    private int numFeatures;
    
    private boolean fitted = false;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private RobustScaler(boolean withCentering, boolean withScaling,
                          double quantileMin, double quantileMax, boolean unitVariance) {
        this.withCentering = withCentering;
        this.withScaling = withScaling;
        this.quantileMin = quantileMin;
        this.quantileMax = quantileMax;
        this.unitVariance = unitVariance;
    }
    
    /**
     * Default constructor with standard settings.
     */
    public RobustScaler() {
        this(true, true, 25.0, 75.0, false);
    }
    
    /**
     * Constructor with centering and scaling options.
     * 
     * @param withCentering Whether to center the data (subtract median)
     * @param withScaling Whether to scale the data (divide by IQR)
     */
    public RobustScaler(boolean withCentering, boolean withScaling) {
        this(withCentering, withScaling, 25.0, 75.0, false);
    }
    
    /**
     * Fits the scaler to the training data.
     * Computes the median and IQR for each feature.
     * 
     * @param X Training data of shape (n_samples, n_features)
     */
    public void fit(double[][] X) {
        validateInput(X);
        
        int nSamples = X.length;
        numFeatures = X[0].length;
        
        center = new double[numFeatures];
        scale = new double[numFeatures];
        
        // For each feature
        for (int j = 0; j < numFeatures; j++) {
            // Extract feature column
            double[] column = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                column[i] = X[i][j];
            }
            
            // Compute median (center)
            if (withCentering) {
                center[j] = computePercentile(column, 50.0);
            } else {
                center[j] = 0.0;
            }
            
            // Compute IQR (scale)
            if (withScaling) {
                double q1 = computePercentile(column, quantileMin);
                double q3 = computePercentile(column, quantileMax);
                double iqr = q3 - q1;
                
                if (unitVariance) {
                    // Adjust scale to achieve unit variance
                    // IQR = 2 * Phi^(-1)(0.75) * sigma for normal distribution
                    // Phi^(-1)(0.75) ≈ 0.6745
                    double factor = 2 * 0.6745;
                    scale[j] = iqr / factor;
                } else {
                    scale[j] = iqr;
                }
                
                // Avoid division by zero
                if (scale[j] == 0.0) {
                    scale[j] = 1.0;
                }
            } else {
                scale[j] = 1.0;
            }
        }
        
        fitted = true;
    }
    
    /**
     * Validates input data.
     */
    private void validateInput(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        
        int nFeatures = X[0].length;
        for (int i = 0; i < X.length; i++) {
            if (X[i] == null) {
                throw new IllegalArgumentException("Row " + i + " is null");
            }
            if (X[i].length != nFeatures) {
                throw new IllegalArgumentException(
                    "All rows must have the same number of features"
                );
            }
        }
    }
    
    /**
     * Computes the percentile of an array.
     * 
     * @param data The data array
     * @param percentile The percentile (0-100)
     * @return The percentile value
     */
    private double computePercentile(double[] data, double percentile) {
        if (data.length == 0) {
            throw new IllegalArgumentException("Cannot compute percentile of empty array");
        }
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("Percentile must be between 0 and 100");
        }
        
        // Sort a copy of the data
        double[] sorted = Arrays.copyOf(data, data.length);
        Arrays.sort(sorted);
        
        // Compute the percentile using linear interpolation
        double index = (percentile / 100.0) * (sorted.length - 1);
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);
        
        if (lower == upper) {
            return sorted[lower];
        }
        
        double weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
    
    /**
     * Transforms the data using the fitted parameters.
     * 
     * @param X Data to transform
     * @return Transformed data
     */
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transformation");
        }
        validateInput(X);
        
        if (X[0].length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, X[0].length)
            );
        }
        
        int nSamples = X.length;
        double[][] result = new double[nSamples][numFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                result[i][j] = (X[i][j] - center[j]) / scale[j];
            }
        }
        
        return result;
    }
    
    /**
     * Fits the scaler and transforms the data in one step.
     * 
     * @param X Data to fit and transform
     * @return Transformed data
     */
    public double[][] fitTransform(double[][] X) {
        fit(X);
        return transform(X);
    }
    
    /**
     * Reverses the transformation.
     * 
     * @param X Scaled data
     * @return Original scale data
     */
    public double[][] inverseTransform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before inverse transformation");
        }
        validateInput(X);
        
        if (X[0].length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, X[0].length)
            );
        }
        
        int nSamples = X.length;
        double[][] result = new double[nSamples][numFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                result[i][j] = X[i][j] * scale[j] + center[j];
            }
        }
        
        return result;
    }
    
    /**
     * Transforms a single sample.
     * 
     * @param x Sample to transform
     * @return Transformed sample
     */
    public double[] transform(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before transformation");
        }
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double[] result = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            result[j] = (x[j] - center[j]) / scale[j];
        }
        
        return result;
    }
    
    /**
     * Reverses the transformation for a single sample.
     * 
     * @param x Scaled sample
     * @return Original scale sample
     */
    public double[] inverseTransform(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Scaler must be fitted before inverse transformation");
        }
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double[] result = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            result[j] = x[j] * scale[j] + center[j];
        }
        
        return result;
    }
    
    /**
     * Returns the center (median) values.
     * 
     * @return Center values for each feature
     */
    public double[] getCenter() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted");
        }
        return Arrays.copyOf(center, center.length);
    }
    
    /**
     * Returns the scale (IQR) values.
     * 
     * @return Scale values for each feature
     */
    public double[] getScale() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted");
        }
        return Arrays.copyOf(scale, scale.length);
    }
    
    /**
     * Returns the quantile range used.
     * 
     * @return Array of [quantileMin, quantileMax]
     */
    public double[] getQuantileRange() {
        return new double[] { quantileMin, quantileMax };
    }
    
    /**
     * Returns whether centering is applied.
     * 
     * @return true if centering is applied
     */
    public boolean isWithCentering() {
        return withCentering;
    }
    
    /**
     * Returns whether scaling is applied.
     * 
     * @return true if scaling is applied
     */
    public boolean isWithScaling() {
        return withScaling;
    }
    
    /**
     * Returns whether the scaler has been fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Returns the number of features.
     * 
     * @return Number of features
     */
    public int getNumFeatures() {
        if (!fitted) {
            throw new IllegalStateException("Scaler has not been fitted");
        }
        return numFeatures;
    }
    
    /**
     * Builder class for RobustScaler.
     */
    public static class Builder {
        private boolean withCentering = true;
        private boolean withScaling = true;
        private double quantileMin = 25.0;
        private double quantileMax = 75.0;
        private boolean unitVariance = false;
        
        /**
         * Sets whether to center the data (subtract median).
         * 
         * @param withCentering Whether to center
         * @return This builder
         */
        public Builder withCentering(boolean withCentering) {
            this.withCentering = withCentering;
            return this;
        }
        
        /**
         * Sets whether to scale the data (divide by IQR).
         * 
         * @param withScaling Whether to scale
         * @return This builder
         */
        public Builder withScaling(boolean withScaling) {
            this.withScaling = withScaling;
            return this;
        }
        
        /**
         * Sets the quantile range.
         * Default is (25.0, 75.0) for the interquartile range (IQR).
         * 
         * @param min Lower quantile (e.g., 25.0)
         * @param max Upper quantile (e.g., 75.0)
         * @return This builder
         */
        public Builder quantileRange(double min, double max) {
            if (min < 0 || min > 100) {
                throw new IllegalArgumentException("quantileMin must be between 0 and 100");
            }
            if (max < 0 || max > 100) {
                throw new IllegalArgumentException("quantileMax must be between 0 and 100");
            }
            if (min >= max) {
                throw new IllegalArgumentException("quantileMin must be less than quantileMax");
            }
            this.quantileMin = min;
            this.quantileMax = max;
            return this;
        }
        
        /**
         * Sets whether to scale to unit variance.
         * When true, the IQR is adjusted to achieve approximately unit variance
         * for normally distributed data.
         * 
         * @param unitVariance Whether to achieve unit variance
         * @return This builder
         */
        public Builder unitVariance(boolean unitVariance) {
            this.unitVariance = unitVariance;
            return this;
        }
        
        /**
         * Builds the RobustScaler instance.
         * 
         * @return A new RobustScaler instance
         */
        public RobustScaler build() {
            return new RobustScaler(withCentering, withScaling, 
                                     quantileMin, quantileMax, unitVariance);
        }
    }
}
