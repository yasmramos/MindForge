package io.github.yasmramos.mindforge.interpret;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;

/**
 * LIME (Local Interpretable Model-agnostic Explanations).
 * Explains individual predictions by approximating the model locally with an interpretable model.
 */
public class LIME implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nSamples;
    private final double kernelWidth;
    private final long seed;
    private double[] featureStd;
    
    private LIME(Builder builder) {
        this.nSamples = builder.nSamples;
        this.kernelWidth = builder.kernelWidth;
        this.seed = builder.seed;
    }
    
    /**
     * Set feature standard deviations for perturbation scaling.
     * @param std Standard deviation for each feature
     */
    public void setFeatureStd(double[] std) {
        this.featureStd = std;
    }
    
    /**
     * Calculate feature standard deviations from data.
     * @param data Training data
     */
    public void fitFeatureStd(double[][] data) {
        int nFeatures = data[0].length;
        featureStd = new double[nFeatures];
        
        for (int j = 0; j < nFeatures; j++) {
            double mean = 0;
            for (double[] row : data) {
                mean += row[j];
            }
            mean /= data.length;
            
            double variance = 0;
            for (double[] row : data) {
                variance += (row[j] - mean) * (row[j] - mean);
            }
            featureStd[j] = Math.sqrt(variance / data.length);
            if (featureStd[j] < 1e-10) featureStd[j] = 1.0;
        }
    }
    
    /**
     * Explain a prediction using LIME.
     * @param predict Prediction function
     * @param instance Instance to explain
     * @return Explanation containing feature weights
     */
    public Explanation explain(Function<double[], Double> predict, double[] instance) {
        if (featureStd == null) {
            throw new IllegalStateException("Feature standard deviations must be set. Call fitFeatureStd() first.");
        }
        
        int nFeatures = instance.length;
        Random random = new Random(seed);
        
        double[][] perturbedData = new double[nSamples][nFeatures];
        double[] predictions = new double[nSamples];
        double[] weights = new double[nSamples];
        
        // Generate perturbed samples
        for (int i = 0; i < nSamples; i++) {
            double distance = 0;
            for (int j = 0; j < nFeatures; j++) {
                double perturbation = random.nextGaussian() * featureStd[j];
                perturbedData[i][j] = instance[j] + perturbation;
                distance += (perturbation / featureStd[j]) * (perturbation / featureStd[j]);
            }
            
            predictions[i] = predict.apply(perturbedData[i]);
            
            // Exponential kernel weight
            weights[i] = Math.exp(-distance / (2 * kernelWidth * kernelWidth));
        }
        
        // Fit weighted linear regression
        double[] coefficients = fitWeightedLinearRegression(perturbedData, predictions, weights, instance);
        
        // Calculate intercept
        double intercept = predict.apply(instance);
        for (int j = 0; j < nFeatures; j++) {
            intercept -= coefficients[j] * instance[j];
        }
        
        return new Explanation(coefficients, intercept, predict.apply(instance));
    }
    
    private double[] fitWeightedLinearRegression(double[][] X, double[] y, double[] weights, double[] center) {
        int n = X.length;
        int p = X[0].length;
        
        // Center the features
        double[][] Xc = new double[n][p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                Xc[i][j] = X[i][j] - center[j];
            }
        }
        
        // Weighted mean of y
        double yMean = 0, wSum = 0;
        for (int i = 0; i < n; i++) {
            yMean += weights[i] * y[i];
            wSum += weights[i];
        }
        yMean /= wSum;
        
        // Weighted least squares
        double[][] XtWX = new double[p][p];
        double[] XtWy = new double[p];
        
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += Xc[k][i] * weights[k] * Xc[k][j];
                }
                XtWX[i][j] = sum;
            }
            
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += Xc[k][i] * weights[k] * (y[k] - yMean);
            }
            XtWy[i] = sum;
        }
        
        // Regularization
        for (int i = 0; i < p; i++) {
            XtWX[i][i] += 1e-6;
        }
        
        return solveLinearSystem(XtWX, XtWy);
    }
    
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = A.length;
        double[][] augmented = new double[n][n + 1];
        
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, augmented[i], 0, n);
            augmented[i][n] = b[i];
        }
        
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
    
    /**
     * Container for LIME explanation results.
     */
    public static class Explanation implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final double[] featureWeights;
        private final double intercept;
        private final double prediction;
        
        public Explanation(double[] featureWeights, double intercept, double prediction) {
            this.featureWeights = featureWeights;
            this.intercept = intercept;
            this.prediction = prediction;
        }
        
        public double[] getFeatureWeights() { return featureWeights.clone(); }
        public double getIntercept() { return intercept; }
        public double getPrediction() { return prediction; }
        
        /**
         * Get top contributing features.
         * @param n Number of top features
         * @return Indices of top features sorted by absolute weight
         */
        public int[] getTopFeatures(int n) {
            Integer[] indices = new Integer[featureWeights.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            
            Arrays.sort(indices, (a, b) -> 
                Double.compare(Math.abs(featureWeights[b]), Math.abs(featureWeights[a])));
            
            int[] result = new int[Math.min(n, indices.length)];
            for (int i = 0; i < result.length; i++) {
                result[i] = indices[i];
            }
            return result;
        }
    }
    
    public static class Builder {
        private int nSamples = 1000;
        private double kernelWidth = 0.75;
        private long seed = 42;
        
        public Builder nSamples(int nSamples) { this.nSamples = nSamples; return this; }
        public Builder kernelWidth(double width) { this.kernelWidth = width; return this; }
        public Builder seed(long seed) { this.seed = seed; return this; }
        
        public LIME build() { return new LIME(this); }
    }
}
