package io.github.yasmramos.mindforge.pipeline;

import java.io.Serializable;
import java.util.*;

/**
 * Column Transformer.
 * 
 * Applies different transformers to different subsets of columns.
 * Useful for heterogeneous data with mixed feature types.
 * 
 * @author MindForge
 */
public class ColumnTransformer implements Pipeline.Transformer, Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Specification for a transformer applied to specific columns.
     */
    public static class TransformerSpec implements Serializable {
        private static final long serialVersionUID = 1L;
        public final String name;
        public final Pipeline.Transformer transformer;
        public final int[] columns;
        
        public TransformerSpec(String name, Pipeline.Transformer transformer, int[] columns) {
            this.name = name;
            this.transformer = transformer;
            this.columns = columns;
        }
    }
    
    private List<TransformerSpec> transformers;
    private String remainder; // "drop" or "passthrough"
    private int[] remainderColumns;
    private boolean fitted;
    private int nInputFeatures;
    private int nOutputFeatures;
    
    /**
     * Creates a ColumnTransformer with default parameters.
     */
    public ColumnTransformer() {
        this(new ArrayList<>(), "drop");
    }
    
    /**
     * Creates a ColumnTransformer with specified transformers.
     */
    public ColumnTransformer(List<TransformerSpec> transformers) {
        this(transformers, "drop");
    }
    
    /**
     * Creates a ColumnTransformer with full configuration.
     * 
     * @param transformers List of transformer specifications
     * @param remainder How to handle remaining columns ("drop" or "passthrough")
     */
    public ColumnTransformer(List<TransformerSpec> transformers, String remainder) {
        this.transformers = new ArrayList<>(transformers);
        this.remainder = remainder;
        this.fitted = false;
    }
    
    /**
     * Builder pattern for ColumnTransformer.
     */
    public static class Builder {
        private List<TransformerSpec> transformers = new ArrayList<>();
        private String remainder = "drop";
        
        public Builder addTransformer(String name, Pipeline.Transformer transformer, int... columns) {
            transformers.add(new TransformerSpec(name, transformer, columns));
            return this;
        }
        
        public Builder remainder(String remainder) {
            this.remainder = remainder;
            return this;
        }
        
        public ColumnTransformer build() {
            return new ColumnTransformer(transformers, remainder);
        }
    }
    
    /**
     * Adds a transformer for specific columns.
     */
    public ColumnTransformer addTransformer(String name, Pipeline.Transformer transformer, int... columns) {
        transformers.add(new TransformerSpec(name, transformer, columns));
        return this;
    }
    
    @Override
    public void fit(double[][] X, int[] y) {
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        
        nInputFeatures = X[0].length;
        
        // Track which columns are used
        Set<Integer> usedColumns = new HashSet<>();
        for (TransformerSpec spec : transformers) {
            for (int col : spec.columns) {
                usedColumns.add(col);
            }
        }
        
        // Find remainder columns
        List<Integer> remainderList = new ArrayList<>();
        for (int i = 0; i < nInputFeatures; i++) {
            if (!usedColumns.contains(i)) {
                remainderList.add(i);
            }
        }
        remainderColumns = remainderList.stream().mapToInt(Integer::intValue).toArray();
        
        // Fit each transformer
        nOutputFeatures = 0;
        
        for (TransformerSpec spec : transformers) {
            double[][] subset = extractColumns(X, spec.columns);
            spec.transformer.fit(subset, y);
            
            // Estimate output features (assuming same as input for now)
            double[][] transformed = spec.transformer.transform(subset);
            nOutputFeatures += transformed[0].length;
        }
        
        if (remainder.equals("passthrough")) {
            nOutputFeatures += remainderColumns.length;
        }
        
        fitted = true;
    }
    
    @Override
    public double[][] transform(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("ColumnTransformer not fitted");
        }
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        
        int n = X.length;
        List<double[][]> transformedParts = new ArrayList<>();
        
        // Transform each subset
        for (TransformerSpec spec : transformers) {
            double[][] subset = extractColumns(X, spec.columns);
            double[][] transformed = spec.transformer.transform(subset);
            transformedParts.add(transformed);
        }
        
        // Add remainder if passthrough
        if (remainder.equals("passthrough") && remainderColumns.length > 0) {
            double[][] remainderData = extractColumns(X, remainderColumns);
            transformedParts.add(remainderData);
        }
        
        // Concatenate results
        return concatenateHorizontally(transformedParts, n);
    }
    
    @Override
    public double[][] fitTransform(double[][] X, int[] y) {
        fit(X, y);
        return transform(X);
    }
    
    /**
     * Extracts specified columns from data.
     */
    private double[][] extractColumns(double[][] X, int[] columns) {
        int n = X.length;
        double[][] subset = new double[n][columns.length];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < columns.length; j++) {
                subset[i][j] = X[i][columns[j]];
            }
        }
        
        return subset;
    }
    
    /**
     * Concatenates arrays horizontally.
     */
    private double[][] concatenateHorizontally(List<double[][]> parts, int nRows) {
        if (parts.isEmpty()) {
            return new double[nRows][0];
        }
        
        int totalCols = 0;
        for (double[][] part : parts) {
            if (part.length > 0) {
                totalCols += part[0].length;
            }
        }
        
        double[][] result = new double[nRows][totalCols];
        
        int colOffset = 0;
        for (double[][] part : parts) {
            if (part.length > 0) {
                int partCols = part[0].length;
                for (int i = 0; i < nRows; i++) {
                    System.arraycopy(part[i], 0, result[i], colOffset, partCols);
                }
                colOffset += partCols;
            }
        }
        
        return result;
    }
    
    /**
     * Gets a transformer by name.
     */
    public Pipeline.Transformer getTransformer(String name) {
        for (TransformerSpec spec : transformers) {
            if (spec.name.equals(name)) {
                return spec.transformer;
            }
        }
        return null;
    }
    
    // Getters
    public boolean isFitted() {
        return fitted;
    }
    
    public int getNumTransformers() {
        return transformers.size();
    }
    
    public int getNumOutputFeatures() {
        return nOutputFeatures;
    }
    
    public String getRemainder() {
        return remainder;
    }
    
    public List<String> getTransformerNames() {
        List<String> names = new ArrayList<>();
        for (TransformerSpec spec : transformers) {
            names.add(spec.name);
        }
        return names;
    }
}
