package io.github.yasmramos.mindforge.decomposition;

import java.util.Arrays;

/**
 * Principal Component Analysis (PCA) for dimensionality reduction.
 * 
 * PCA uses Singular Value Decomposition (SVD) to project data to a lower 
 * dimensional space while preserving as much variance as possible.
 * 
 * The implementation centers the data but does not scale it. For data with
 * features on different scales, consider using StandardScaler before PCA.
 * 
 * Example usage:
 * <pre>
 * // Reduce to 2 components
 * PCA pca = new PCA(2);
 * pca.fit(X);
 * double[][] X_reduced = pca.transform(X);
 * 
 * // Or in one step
 * double[][] X_reduced = pca.fitTransform(X);
 * 
 * // Reconstruct original data (with some loss)
 * double[][] X_reconstructed = pca.inverseTransform(X_reduced);
 * </pre>
 * 
 * @author MindForge
 * @version 1.0.8-alpha
 */
public class PCA {
    
    private final int nComponents;
    private double[] mean;
    private double[][] components;
    private double[] explainedVariance;
    private double[] explainedVarianceRatio;
    private double[] singularValues;
    private int nFeaturesIn;
    private int nSamples;
    private boolean fitted;
    
    /**
     * Creates a PCA transformer with the specified number of components.
     * 
     * @param nComponents Number of principal components to keep.
     *                    If -1, keeps all components.
     * @throws IllegalArgumentException if nComponents is less than 1 and not -1
     */
    public PCA(int nComponents) {
        if (nComponents < 1 && nComponents != -1) {
            throw new IllegalArgumentException(
                "nComponents must be positive or -1 for all, got: " + nComponents);
        }
        this.nComponents = nComponents;
        this.fitted = false;
    }
    
    /**
     * Creates a PCA transformer that keeps all components.
     */
    public PCA() {
        this(-1);
    }
    
    /**
     * Fits the PCA model to the training data.
     * 
     * @param X Training data of shape [n_samples, n_features]
     * @return this PCA instance for method chaining
     */
    public PCA fit(double[][] X) {
        validateInput(X);
        
        nSamples = X.length;
        nFeaturesIn = X[0].length;
        int actualComponents = (nComponents == -1) ? 
            Math.min(nSamples, nFeaturesIn) : Math.min(nComponents, Math.min(nSamples, nFeaturesIn));
        
        // Center the data
        mean = new double[nFeaturesIn];
        for (int j = 0; j < nFeaturesIn; j++) {
            for (int i = 0; i < nSamples; i++) {
                mean[j] += X[i][j];
            }
            mean[j] /= nSamples;
        }
        
        // Create centered matrix
        double[][] centered = new double[nSamples][nFeaturesIn];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeaturesIn; j++) {
                centered[i][j] = X[i][j] - mean[j];
            }
        }
        
        // Perform SVD using power iteration method
        SVDResult svd = computeSVD(centered, actualComponents);
        
        // Store results
        components = svd.vt;
        singularValues = svd.s;
        
        // Calculate explained variance
        explainedVariance = new double[actualComponents];
        double totalVariance = 0.0;
        for (int i = 0; i < actualComponents; i++) {
            explainedVariance[i] = (singularValues[i] * singularValues[i]) / (nSamples - 1);
            totalVariance += explainedVariance[i];
        }
        
        // Calculate total variance (including all components)
        double fullVariance = 0.0;
        for (int j = 0; j < nFeaturesIn; j++) {
            double colVar = 0.0;
            for (int i = 0; i < nSamples; i++) {
                colVar += centered[i][j] * centered[i][j];
            }
            fullVariance += colVar / (nSamples - 1);
        }
        
        // Calculate explained variance ratio
        explainedVarianceRatio = new double[actualComponents];
        for (int i = 0; i < actualComponents; i++) {
            explainedVarianceRatio[i] = explainedVariance[i] / fullVariance;
        }
        
        fitted = true;
        return this;
    }
    
    /**
     * Transforms the data to the principal component space.
     * 
     * @param X Data to transform of shape [n_samples, n_features]
     * @return Transformed data of shape [n_samples, n_components]
     */
    public double[][] transform(double[][] X) {
        checkFitted();
        validateInput(X);
        
        if (X[0].length != nFeaturesIn) {
            throw new IllegalArgumentException(
                "X has " + X[0].length + " features, but PCA was fitted with " + 
                nFeaturesIn + " features");
        }
        
        int n = X.length;
        int nComp = components.length;
        double[][] result = new double[n][nComp];
        
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < nComp; k++) {
                double sum = 0.0;
                for (int j = 0; j < nFeaturesIn; j++) {
                    sum += (X[i][j] - mean[j]) * components[k][j];
                }
                result[i][k] = sum;
            }
        }
        
        return result;
    }
    
    /**
     * Fits the model and transforms the data in one step.
     * 
     * @param X Training data of shape [n_samples, n_features]
     * @return Transformed data of shape [n_samples, n_components]
     */
    public double[][] fitTransform(double[][] X) {
        return fit(X).transform(X);
    }
    
    /**
     * Transforms data back to original space.
     * This reconstructs the original data from the reduced representation.
     * 
     * @param X_transformed Transformed data of shape [n_samples, n_components]
     * @return Reconstructed data of shape [n_samples, n_features]
     */
    public double[][] inverseTransform(double[][] X_transformed) {
        checkFitted();
        
        if (X_transformed == null || X_transformed.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        if (X_transformed[0].length != components.length) {
            throw new IllegalArgumentException(
                "X_transformed has " + X_transformed[0].length + " features, expected " + 
                components.length);
        }
        
        int n = X_transformed.length;
        int nComp = components.length;
        double[][] result = new double[n][nFeaturesIn];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nFeaturesIn; j++) {
                double sum = mean[j];
                for (int k = 0; k < nComp; k++) {
                    sum += X_transformed[i][k] * components[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        return result;
    }
    
    /**
     * Simplified SVD computation using power iteration.
     * Computes the top k singular values and vectors.
     */
    private SVDResult computeSVD(double[][] A, int k) {
        int m = A.length;
        int n = A[0].length;
        
        double[] s = new double[k];
        double[][] vt = new double[k][n];
        
        // Create working copy
        double[][] work = new double[m][n];
        for (int i = 0; i < m; i++) {
            work[i] = Arrays.copyOf(A[i], n);
        }
        
        for (int comp = 0; comp < k; comp++) {
            // Power iteration to find largest singular value/vector
            double[] v = new double[n];
            // Initialize with random vector
            java.util.Random random = new java.util.Random(42 + comp);
            for (int i = 0; i < n; i++) {
                v[i] = random.nextGaussian();
            }
            normalize(v);
            
            // Power iteration
            int maxIter = 100;
            for (int iter = 0; iter < maxIter; iter++) {
                // u = A * v
                double[] u = new double[m];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        u[i] += work[i][j] * v[j];
                    }
                }
                
                // v = A^T * u
                double[] vNew = new double[n];
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < m; i++) {
                        vNew[j] += work[i][j] * u[i];
                    }
                }
                
                normalize(vNew);
                
                // Check convergence
                double diff = 0.0;
                for (int i = 0; i < n; i++) {
                    diff += Math.abs(vNew[i] - v[i]);
                }
                
                v = vNew;
                
                if (diff < 1e-10) {
                    break;
                }
            }
            
            // Calculate singular value
            double[] u = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    u[i] += work[i][j] * v[j];
                }
            }
            
            double sigma = 0.0;
            for (int i = 0; i < m; i++) {
                sigma += u[i] * u[i];
            }
            sigma = Math.sqrt(sigma);
            
            s[comp] = sigma;
            vt[comp] = v;
            
            // Deflate: remove the contribution of this component
            if (sigma > 1e-10) {
                for (int i = 0; i < m; i++) {
                    u[i] /= sigma;
                }
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        work[i][j] -= sigma * u[i] * v[j];
                    }
                }
            }
        }
        
        return new SVDResult(s, vt);
    }
    
    private void normalize(double[] v) {
        double norm = 0.0;
        for (double val : v) {
            norm += val * val;
        }
        norm = Math.sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < v.length; i++) {
                v[i] /= norm;
            }
        }
    }
    
    private static class SVDResult {
        final double[] s;
        final double[][] vt;
        
        SVDResult(double[] s, double[][] vt) {
            this.s = s;
            this.vt = vt;
        }
    }
    
    /**
     * Gets the principal components (eigenvectors of the covariance matrix).
     * Each row represents a principal component.
     * 
     * @return Components array of shape [n_components, n_features]
     */
    public double[][] getComponents() {
        checkFitted();
        double[][] result = new double[components.length][];
        for (int i = 0; i < components.length; i++) {
            result[i] = Arrays.copyOf(components[i], components[i].length);
        }
        return result;
    }
    
    /**
     * Gets the variance explained by each principal component.
     * 
     * @return Array of explained variances
     */
    public double[] getExplainedVariance() {
        checkFitted();
        return Arrays.copyOf(explainedVariance, explainedVariance.length);
    }
    
    /**
     * Gets the percentage of variance explained by each principal component.
     * 
     * @return Array of explained variance ratios (sums to approximately 1.0
     *         when all components are kept)
     */
    public double[] getExplainedVarianceRatio() {
        checkFitted();
        return Arrays.copyOf(explainedVarianceRatio, explainedVarianceRatio.length);
    }
    
    /**
     * Gets the cumulative explained variance ratio.
     * 
     * @return Array where each element is the sum of explained variance ratios
     *         up to and including that component
     */
    public double[] getCumulativeExplainedVarianceRatio() {
        checkFitted();
        double[] cumulative = new double[explainedVarianceRatio.length];
        double sum = 0.0;
        for (int i = 0; i < cumulative.length; i++) {
            sum += explainedVarianceRatio[i];
            cumulative[i] = sum;
        }
        return cumulative;
    }
    
    /**
     * Gets the singular values from the SVD decomposition.
     * 
     * @return Array of singular values in decreasing order
     */
    public double[] getSingularValues() {
        checkFitted();
        return Arrays.copyOf(singularValues, singularValues.length);
    }
    
    /**
     * Gets the mean of each feature computed during fit.
     * 
     * @return Array of feature means
     */
    public double[] getMean() {
        checkFitted();
        return Arrays.copyOf(mean, mean.length);
    }
    
    /**
     * Gets the number of components.
     * 
     * @return Number of components (after fitting)
     */
    public int getNumberOfComponents() {
        checkFitted();
        return components.length;
    }
    
    /**
     * Gets the number of features seen during fit.
     * 
     * @return Number of input features
     */
    public int getNumberOfFeatures() {
        checkFitted();
        return nFeaturesIn;
    }
    
    /**
     * Gets the requested number of components parameter.
     * 
     * @return The nComponents parameter (-1 means all)
     */
    public int getNComponents() {
        return nComponents;
    }
    
    /**
     * Checks if the PCA model has been fitted.
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
                "This PCA instance is not fitted yet. " +
                "Call 'fit' with appropriate arguments before using this method.");
        }
    }
    
    @Override
    public String toString() {
        if (fitted) {
            return String.format("PCA(n_components=%d, n_features=%d, explained_variance=%.2f%%)",
                components.length, nFeaturesIn, 
                Arrays.stream(explainedVarianceRatio).sum() * 100);
        } else {
            return String.format("PCA(n_components=%s, fitted=false)", 
                nComponents == -1 ? "all" : String.valueOf(nComponents));
        }
    }
}
