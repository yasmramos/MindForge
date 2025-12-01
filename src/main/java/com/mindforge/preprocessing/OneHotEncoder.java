package com.mindforge.preprocessing;

import java.io.Serializable;
import java.util.*;

/**
 * Encode categorical features as a one-hot numeric array.
 * 
 * <p>The input to this transformer should be an array of integers or strings,
 * denoting the values taken on by categorical features. The features are encoded
 * using a one-hot (aka 'one-of-K' or 'dummy') encoding scheme.</p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * OneHotEncoder encoder = new OneHotEncoder();
 * String[][] X = {{"cat"}, {"dog"}, {"cat"}, {"bird"}};
 * double[][] encoded = encoder.fitTransform(X);
 * // Unique categories: [bird, cat, dog]
 * // encoded = [[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
 * }</pre>
 * 
 * @author Matrix Agent
 * @version 1.0
 */
public class OneHotEncoder implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final boolean dropFirst;
    private final boolean handleUnknown;
    private final String unknownValue;
    private List<Map<String, Integer>> categoryMaps;
    private List<String[]> categories;
    private int nFeatures;
    private int nOutputFeatures;
    private boolean isFitted;
    
    /**
     * Creates a OneHotEncoder with default settings.
     */
    public OneHotEncoder() {
        this(false, false, null);
    }
    
    /**
     * Creates a OneHotEncoder with specified settings.
     * 
     * @param dropFirst If true, drop the first category in each feature to avoid multicollinearity
     */
    public OneHotEncoder(boolean dropFirst) {
        this(dropFirst, false, null);
    }
    
    /**
     * Creates a OneHotEncoder with full configuration.
     * 
     * @param dropFirst If true, drop the first category in each feature
     * @param handleUnknown If true, handle unknown categories during transform
     * @param unknownValue Value to use for unknown categories (null means all zeros)
     */
    public OneHotEncoder(boolean dropFirst, boolean handleUnknown, String unknownValue) {
        this.dropFirst = dropFirst;
        this.handleUnknown = handleUnknown;
        this.unknownValue = unknownValue;
        this.isFitted = false;
    }
    
    /**
     * Fit the encoder to the data.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return this encoder
     * @throws IllegalArgumentException if X is null or empty
     */
    public OneHotEncoder fit(String[][] X) {
        validateInput(X);
        
        this.nFeatures = X[0].length;
        this.categoryMaps = new ArrayList<>();
        this.categories = new ArrayList<>();
        
        // For each feature, find unique categories
        for (int j = 0; j < nFeatures; j++) {
            Set<String> uniqueCategories = new TreeSet<>(); // TreeSet for consistent ordering
            for (String[] row : X) {
                if (row[j] != null) {
                    uniqueCategories.add(row[j]);
                }
            }
            
            String[] cats = uniqueCategories.toArray(new String[0]);
            categories.add(cats);
            
            Map<String, Integer> catMap = new HashMap<>();
            for (int i = 0; i < cats.length; i++) {
                catMap.put(cats[i], i);
            }
            categoryMaps.add(catMap);
        }
        
        // Calculate output features
        this.nOutputFeatures = 0;
        for (String[] cats : categories) {
            nOutputFeatures += dropFirst ? cats.length - 1 : cats.length;
        }
        
        this.isFitted = true;
        return this;
    }
    
    /**
     * Fit the encoder to integer data.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return this encoder
     */
    public OneHotEncoder fit(int[][] X) {
        return fit(intToString(X));
    }
    
    /**
     * Transform categorical data to one-hot encoded format.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return One-hot encoded data of shape [n_samples, n_output_features]
     * @throws IllegalStateException if the encoder has not been fitted
     * @throws IllegalArgumentException if X has different number of features or contains unknown categories
     */
    public double[][] transform(String[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("OneHotEncoder must be fitted before transform");
        }
        validateInput(X);
        
        if (X[0].length != nFeatures) {
            throw new IllegalArgumentException(
                String.format("X has %d features, but OneHotEncoder is expecting %d features",
                    X[0].length, nFeatures));
        }
        
        int nSamples = X.length;
        double[][] result = new double[nSamples][nOutputFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            int outputIdx = 0;
            for (int j = 0; j < nFeatures; j++) {
                String value = X[i][j];
                Map<String, Integer> catMap = categoryMaps.get(j);
                int nCats = categories.get(j).length;
                int encodedSize = dropFirst ? nCats - 1 : nCats;
                
                if (value == null) {
                    // Handle null as all zeros
                    outputIdx += encodedSize;
                    continue;
                }
                
                Integer catIdx = catMap.get(value);
                
                if (catIdx == null) {
                    if (!handleUnknown) {
                        throw new IllegalArgumentException(
                            String.format("Unknown category '%s' found in feature %d", value, j));
                    }
                    // Handle unknown - leave as zeros or use unknownValue if specified
                    if (unknownValue != null) {
                        catIdx = catMap.get(unknownValue);
                    }
                }
                
                if (catIdx != null) {
                    int adjustedIdx = dropFirst ? catIdx - 1 : catIdx;
                    if (adjustedIdx >= 0 && adjustedIdx < encodedSize) {
                        result[i][outputIdx + adjustedIdx] = 1.0;
                    }
                }
                
                outputIdx += encodedSize;
            }
        }
        
        return result;
    }
    
    /**
     * Transform integer categorical data.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return One-hot encoded data
     */
    public double[][] transform(int[][] X) {
        return transform(intToString(X));
    }
    
    /**
     * Fit to data, then transform it.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return One-hot encoded data
     */
    public double[][] fitTransform(String[][] X) {
        fit(X);
        return transform(X);
    }
    
    /**
     * Fit to integer data, then transform it.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return One-hot encoded data
     */
    public double[][] fitTransform(int[][] X) {
        return fitTransform(intToString(X));
    }
    
    /**
     * Inverse transform one-hot encoded data back to categorical.
     * 
     * @param X One-hot encoded data
     * @return Original categorical data
     * @throws IllegalStateException if not fitted
     */
    public String[][] inverseTransform(double[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("OneHotEncoder must be fitted before inverse transform");
        }
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        
        if (X[0].length != nOutputFeatures) {
            throw new IllegalArgumentException(
                String.format("X has %d features, but expected %d features",
                    X[0].length, nOutputFeatures));
        }
        
        int nSamples = X.length;
        String[][] result = new String[nSamples][nFeatures];
        
        for (int i = 0; i < nSamples; i++) {
            int inputIdx = 0;
            for (int j = 0; j < nFeatures; j++) {
                String[] cats = categories.get(j);
                int encodedSize = dropFirst ? cats.length - 1 : cats.length;
                
                int maxIdx = -1;
                double maxVal = -1;
                
                for (int k = 0; k < encodedSize; k++) {
                    if (X[i][inputIdx + k] > maxVal) {
                        maxVal = X[i][inputIdx + k];
                        maxIdx = k;
                    }
                }
                
                if (maxVal > 0) {
                    int catIdx = dropFirst ? maxIdx + 1 : maxIdx;
                    result[i][j] = cats[catIdx];
                } else if (dropFirst) {
                    result[i][j] = cats[0]; // First category when dropFirst and all zeros
                }
                
                inputIdx += encodedSize;
            }
        }
        
        return result;
    }
    
    /**
     * Convert integer array to string array.
     */
    private String[][] intToString(int[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        String[][] result = new String[X.length][X[0].length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                result[i][j] = String.valueOf(X[i][j]);
            }
        }
        return result;
    }
    
    /**
     * Validate input data.
     */
    private void validateInput(String[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        if (X[0] == null || X[0].length == 0) {
            throw new IllegalArgumentException("Input features cannot be null or empty");
        }
    }
    
    /**
     * Get categories for each feature.
     * 
     * @return list of category arrays for each feature
     * @throws IllegalStateException if not fitted
     */
    public List<String[]> getCategories() {
        if (!isFitted) {
            throw new IllegalStateException("OneHotEncoder must be fitted first");
        }
        return new ArrayList<>(categories);
    }
    
    /**
     * Get the number of input features.
     * 
     * @return number of input features
     * @throws IllegalStateException if not fitted
     */
    public int getNFeatures() {
        if (!isFitted) {
            throw new IllegalStateException("OneHotEncoder must be fitted first");
        }
        return nFeatures;
    }
    
    /**
     * Get the number of output features.
     * 
     * @return number of output features after encoding
     * @throws IllegalStateException if not fitted
     */
    public int getNOutputFeatures() {
        if (!isFitted) {
            throw new IllegalStateException("OneHotEncoder must be fitted first");
        }
        return nOutputFeatures;
    }
    
    /**
     * Check if the encoder drops the first category.
     * 
     * @return true if dropFirst is enabled
     */
    public boolean isDropFirst() {
        return dropFirst;
    }
    
    /**
     * Check if unknown categories are handled.
     * 
     * @return true if handleUnknown is enabled
     */
    public boolean isHandleUnknown() {
        return handleUnknown;
    }
    
    /**
     * Check if the encoder has been fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return isFitted;
    }
}
