package io.github.yasmramos.mindforge.preprocessing;

import java.io.Serializable;
import java.util.*;

/**
 * Target Encoder.
 * 
 * Encodes categorical features based on target statistics.
 * For classification: uses probability of each class.
 * For regression: uses mean of target values.
 * 
 * @author MindForge
 */
public class TargetEncoder implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private double smoothing;
    private double minSamplesLeaf;
    private boolean handleUnknown;
    private double fillValue;
    
    private Map<Integer, Map<String, Double>> encodings;
    private double[] globalMeans;
    private boolean fitted;
    private int nFeatures;
    
    /**
     * Creates a TargetEncoder with default parameters.
     */
    public TargetEncoder() {
        this(1.0, 1, true, 0.0);
    }
    
    /**
     * Creates a TargetEncoder with specified smoothing.
     * 
     * @param smoothing Smoothing parameter for regularization
     */
    public TargetEncoder(double smoothing) {
        this(smoothing, 1, true, 0.0);
    }
    
    /**
     * Creates a TargetEncoder with full configuration.
     * 
     * @param smoothing Smoothing parameter
     * @param minSamplesLeaf Minimum samples for a category
     * @param handleUnknown Whether to handle unknown categories
     * @param fillValue Value for unknown categories
     */
    public TargetEncoder(double smoothing, double minSamplesLeaf, 
                         boolean handleUnknown, double fillValue) {
        if (smoothing < 0) {
            throw new IllegalArgumentException("Smoothing must be non-negative");
        }
        if (minSamplesLeaf < 1) {
            throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
        }
        
        this.smoothing = smoothing;
        this.minSamplesLeaf = minSamplesLeaf;
        this.handleUnknown = handleUnknown;
        this.fillValue = fillValue;
        this.fitted = false;
    }
    
    /**
     * Fits the encoder for regression targets.
     * 
     * @param X Categorical features (as strings)
     * @param y Target values
     * @return this encoder
     */
    public TargetEncoder fit(String[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same length");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        
        int n = X.length;
        nFeatures = X[0].length;
        
        // Compute global mean
        double globalMean = 0;
        for (double val : y) {
            globalMean += val;
        }
        globalMean /= n;
        
        globalMeans = new double[nFeatures];
        Arrays.fill(globalMeans, globalMean);
        
        encodings = new HashMap<>();
        
        for (int j = 0; j < nFeatures; j++) {
            Map<String, List<Double>> categoryValues = new HashMap<>();
            
            // Group target values by category
            for (int i = 0; i < n; i++) {
                String category = X[i][j];
                categoryValues.computeIfAbsent(category, k -> new ArrayList<>()).add(y[i]);
            }
            
            // Compute smoothed means
            Map<String, Double> featureEncodings = new HashMap<>();
            
            for (Map.Entry<String, List<Double>> entry : categoryValues.entrySet()) {
                List<Double> values = entry.getValue();
                int count = values.size();
                
                double categoryMean = 0;
                for (double val : values) {
                    categoryMean += val;
                }
                categoryMean /= count;
                
                // Apply smoothing: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                double smoothedMean = (count * categoryMean + smoothing * globalMean) / (count + smoothing);
                featureEncodings.put(entry.getKey(), smoothedMean);
            }
            
            encodings.put(j, featureEncodings);
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Fits the encoder for classification targets.
     * 
     * @param X Categorical features (as strings)
     * @param y Target class labels
     * @return this encoder
     */
    public TargetEncoder fit(String[][] X, int[] y) {
        // Convert int[] to double[] (class probabilities)
        double[] yDouble = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            yDouble[i] = y[i];
        }
        return fit(X, yDouble);
    }
    
    /**
     * Transforms categorical features to encoded values.
     * 
     * @param X Categorical features
     * @return Encoded numeric features
     */
    public double[][] transform(String[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Encoder not fitted");
        }
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        
        int n = X.length;
        double[][] encoded = new double[n][nFeatures];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nFeatures; j++) {
                String category = X[i][j];
                Map<String, Double> featureEncodings = encodings.get(j);
                
                if (featureEncodings.containsKey(category)) {
                    encoded[i][j] = featureEncodings.get(category);
                } else if (handleUnknown) {
                    encoded[i][j] = fillValue != 0 ? fillValue : globalMeans[j];
                } else {
                    throw new IllegalArgumentException("Unknown category: " + category);
                }
            }
        }
        
        return encoded;
    }
    
    /**
     * Fits and transforms in one step.
     */
    public double[][] fitTransform(String[][] X, double[] y) {
        fit(X, y);
        return transform(X);
    }
    
    /**
     * Fits and transforms for classification.
     */
    public double[][] fitTransform(String[][] X, int[] y) {
        fit(X, y);
        return transform(X);
    }
    
    // Getters
    public boolean isFitted() {
        return fitted;
    }
    
    public double getSmoothing() {
        return smoothing;
    }
    
    public int getNumFeatures() {
        return nFeatures;
    }
    
    public double[] getGlobalMeans() {
        return globalMeans != null ? globalMeans.clone() : null;
    }
}
