package io.github.yasmramos.mindforge.interpret;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;

/**
 * SHAP (SHapley Additive exPlanations) for model interpretability.
 * Computes feature importance based on Shapley values from cooperative game theory.
 * 
 * Implements Kernel SHAP, a model-agnostic approximation method.
 */
public class SHAP implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nSamples;
    private final long seed;
    private double[][] backgroundData;
    private double expectedValue;
    
    private SHAP(Builder builder) {
        this.nSamples = builder.nSamples;
        this.seed = builder.seed;
    }
    
    /**
     * Set background data for computing expected values.
     * @param background Representative samples from training data
     */
    public void setBackground(double[][] background) {
        this.backgroundData = background;
    }
    
    /**
     * Compute SHAP values for a single instance.
     * @param predict Prediction function that takes features and returns prediction
     * @param instance The instance to explain
     * @return SHAP values for each feature
     */
    public double[] explain(Function<double[], Double> predict, double[] instance) {
        if (backgroundData == null || backgroundData.length == 0) {
            throw new IllegalStateException("Background data must be set before explaining");
        }
        
        int nFeatures = instance.length;
        Random random = new Random(seed);
        
        // Calculate expected value using background data
        expectedValue = 0;
        for (double[] bg : backgroundData) {
            expectedValue += predict.apply(bg);
        }
        expectedValue /= backgroundData.length;
        
        // Kernel SHAP approximation using sampling
        double[] shapValues = new double[nFeatures];
        double[] weights = new double[nSamples];
        double[][] coalitions = new double[nSamples][nFeatures];
        double[] predictions = new double[nSamples];
        
        for (int s = 0; s < nSamples; s++) {
            // Sample a coalition (subset of features)
            boolean[] coalition = sampleCoalition(nFeatures, random);
            int coalitionSize = countTrue(coalition);
            
            // Calculate Shapley kernel weight
            weights[s] = shapleyKernelWeight(nFeatures, coalitionSize);
            
            // Create masked instance
            double[] maskedInstance = createMaskedInstance(instance, coalition, random);
            
            // Store coalition as binary vector
            for (int j = 0; j < nFeatures; j++) {
                coalitions[s][j] = coalition[j] ? 1.0 : 0.0;
            }
            
            // Get prediction for masked instance
            predictions[s] = predict.apply(maskedInstance);
        }
        
        // Solve weighted linear regression: predictions = coalitions * shapValues + expectedValue
        shapValues = solveWeightedRegression(coalitions, predictions, weights, expectedValue);
        
        return shapValues;
    }
    
    /**
     * Compute SHAP values for multiple instances.
     * @param predict Prediction function
     * @param instances Instances to explain
     * @return SHAP values matrix [n_instances, n_features]
     */
    public double[][] explainBatch(Function<double[], Double> predict, double[][] instances) {
        double[][] shapValues = new double[instances.length][];
        for (int i = 0; i < instances.length; i++) {
            shapValues[i] = explain(predict, instances[i]);
        }
        return shapValues;
    }
    
    /**
     * Compute mean absolute SHAP values (global feature importance).
     * @param predict Prediction function
     * @param instances Instances to analyze
     * @return Mean absolute SHAP value for each feature
     */
    public double[] meanAbsoluteShap(Function<double[], Double> predict, double[][] instances) {
        double[][] allShap = explainBatch(predict, instances);
        int nFeatures = allShap[0].length;
        double[] meanAbs = new double[nFeatures];
        
        for (int j = 0; j < nFeatures; j++) {
            double sum = 0;
            for (double[] shap : allShap) {
                sum += Math.abs(shap[j]);
            }
            meanAbs[j] = sum / allShap.length;
        }
        
        return meanAbs;
    }
    
    private boolean[] sampleCoalition(int nFeatures, Random random) {
        boolean[] coalition = new boolean[nFeatures];
        for (int i = 0; i < nFeatures; i++) {
            coalition[i] = random.nextBoolean();
        }
        return coalition;
    }
    
    private int countTrue(boolean[] arr) {
        int count = 0;
        for (boolean b : arr) if (b) count++;
        return count;
    }
    
    private double shapleyKernelWeight(int M, int s) {
        if (s == 0 || s == M) {
            return 1e6; // Large weight for empty and full coalitions
        }
        // Shapley kernel: (M-1) / (C(M,s) * s * (M-s))
        double binomial = binomialCoefficient(M, s);
        return (M - 1.0) / (binomial * s * (M - s));
    }
    
    private double binomialCoefficient(int n, int k) {
        if (k > n - k) k = n - k;
        double result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
    
    private double[] createMaskedInstance(double[] instance, boolean[] coalition, Random random) {
        double[] masked = new double[instance.length];
        int bgIdx = random.nextInt(backgroundData.length);
        
        for (int i = 0; i < instance.length; i++) {
            if (coalition[i]) {
                masked[i] = instance[i];
            } else {
                masked[i] = backgroundData[bgIdx][i];
            }
        }
        return masked;
    }
    
    private double[] solveWeightedRegression(double[][] X, double[] y, double[] weights, double intercept) {
        int n = X.length;
        int p = X[0].length;
        
        // Adjust y for intercept
        double[] yAdj = new double[n];
        for (int i = 0; i < n; i++) {
            yAdj[i] = y[i] - intercept;
        }
        
        // Weighted least squares: (X'WX)^-1 * X'Wy
        double[][] XtWX = new double[p][p];
        double[] XtWy = new double[p];
        
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += X[k][i] * weights[k] * X[k][j];
                }
                XtWX[i][j] = sum;
            }
            
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += X[k][i] * weights[k] * yAdj[k];
            }
            XtWy[i] = sum;
        }
        
        // Add regularization for numerical stability
        for (int i = 0; i < p; i++) {
            XtWX[i][i] += 1e-6;
        }
        
        // Solve using Gaussian elimination
        return solveLinearSystem(XtWX, XtWy);
    }
    
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = A.length;
        double[][] augmented = new double[n][n + 1];
        
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, augmented[i], 0, n);
            augmented[i][n] = b[i];
        }
        
        // Forward elimination
        for (int i = 0; i < n; i++) {
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            if (Math.abs(augmented[i][i]) < 1e-10) continue;
            
            for (int k = i + 1; k < n; k++) {
                double factor = augmented[k][i] / augmented[i][i];
                for (int j = i; j <= n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            if (Math.abs(augmented[i][i]) > 1e-10) {
                x[i] /= augmented[i][i];
            }
        }
        
        return x;
    }
    
    public double getExpectedValue() { return expectedValue; }
    
    public static class Builder {
        private int nSamples = 100;
        private long seed = 42;
        
        public Builder nSamples(int nSamples) { this.nSamples = nSamples; return this; }
        public Builder seed(long seed) { this.seed = seed; return this; }
        
        public SHAP build() { return new SHAP(this); }
    }
}
