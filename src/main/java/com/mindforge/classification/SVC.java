package com.mindforge.classification;

import java.util.*;

/**
 * Support Vector Classifier (SVC) for binary and multiclass classification.
 * 
 * Implements a simplified version of Support Vector Machines using:
 * - Linear kernel (for now)
 * - One-vs-Rest strategy for multiclass classification
 * - Gradient descent optimization
 * 
 * Future enhancements could include:
 * - RBF, polynomial, and sigmoid kernels
 * - SMO (Sequential Minimal Optimization) algorithm
 * - Kernel trick for non-linear classification
 * 
 * Example usage:
 * <pre>
 * SVC svc = new SVC.Builder()
 *     .C(1.0)
 *     .maxIter(1000)
 *     .tol(1e-3)
 *     .build();
 * 
 * svc.train(X_train, y_train);
 * int[] predictions = svc.predict(X_test);
 * </pre>
 */
public class SVC implements Classifier<double[]> {
    
    private double C; // Regularization parameter
    private int maxIter;
    private double tol; // Tolerance for stopping criterion
    private double learningRate;
    
    private int numClasses;
    private int numFeatures;
    private int[] classes;
    private double[][] weights; // [class][feature]
    private double[] bias; // [class]
    
    private boolean isTrained = false;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private SVC(double C, int maxIter, double tol, double learningRate) {
        this.C = C;
        this.maxIter = maxIter;
        this.tol = tol;
        this.learningRate = learningRate;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        numFeatures = X[0].length;
        
        // Find unique classes
        Set<Integer> classSet = new HashSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        numClasses = classSet.size();
        classes = new int[numClasses];
        int idx = 0;
        for (int cls : classSet) {
            classes[idx++] = cls;
        }
        Arrays.sort(classes);
        
        // Initialize weights and bias
        weights = new double[numClasses][numFeatures];
        bias = new double[numClasses];
        
        // Train one-vs-rest classifiers
        for (int c = 0; c < numClasses; c++) {
            trainBinaryClassifier(X, y, c);
        }
        
        isTrained = true;
    }
    
    /**
     * Trains a binary classifier for one class vs the rest.
     */
    private void trainBinaryClassifier(double[][] X, int[] y, int classIdx) {
        int n = X.length;
        int targetClass = classes[classIdx];
        
        // Create binary labels: +1 for target class, -1 for others
        int[] binaryY = new int[n];
        for (int i = 0; i < n; i++) {
            binaryY[i] = (y[i] == targetClass) ? 1 : -1;
        }
        
        // Initialize weights and bias for this classifier
        double[] w = new double[numFeatures];
        double b = 0.0;
        
        Random random = new Random(42);
        for (int f = 0; f < numFeatures; f++) {
            w[f] = random.nextGaussian() * 0.01;
        }
        
        // Gradient descent optimization
        for (int iter = 0; iter < maxIter; iter++) {
            double[] gradW = new double[numFeatures];
            double gradB = 0.0;
            double loss = 0.0;
            
            // Compute gradients
            for (int i = 0; i < n; i++) {
                double prediction = computeScore(X[i], w, b);
                double margin = binaryY[i] * prediction;
                
                // Hinge loss and gradient
                if (margin < 1) {
                    loss += 1 - margin;
                    
                    // Gradient of hinge loss
                    for (int f = 0; f < numFeatures; f++) {
                        gradW[f] -= binaryY[i] * X[i][f];
                    }
                    gradB -= binaryY[i];
                }
            }
            
            // Add regularization term
            for (int f = 0; f < numFeatures; f++) {
                loss += 0.5 * C * w[f] * w[f];
                gradW[f] += C * w[f];
            }
            
            // Update weights and bias
            for (int f = 0; f < numFeatures; f++) {
                w[f] -= learningRate * gradW[f] / n;
            }
            b -= learningRate * gradB / n;
            
            // Check convergence
            if (iter > 0 && Math.abs(loss) < tol) {
                break;
            }
        }
        
        // Store trained parameters
        weights[classIdx] = w;
        bias[classIdx] = b;
    }
    
    /**
     * Computes the decision score for a sample.
     */
    private double computeScore(double[] x, double[] w, double b) {
        double score = b;
        for (int f = 0; f < x.length; f++) {
            score += w[f] * x[f];
        }
        return score;
    }
    
    @Override
    public int predict(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double maxScore = Double.NEGATIVE_INFINITY;
        int predictedClass = classes[0];
        
        // Find class with maximum decision score
        for (int c = 0; c < numClasses; c++) {
            double score = computeScore(x, weights[c], bias[c]);
            
            if (score > maxScore) {
                maxScore = score;
                predictedClass = classes[c];
            }
        }
        
        return predictedClass;
    }
    
    /**
     * Predicts class labels for multiple samples.
     */
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Computes decision scores for all classes.
     */
    public double[] decisionFunction(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double[] scores = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            scores[c] = computeScore(x, weights[c], bias[c]);
        }
        
        return scores;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Returns the class labels.
     */
    public int[] getClasses() {
        return Arrays.copyOf(classes, classes.length);
    }
    
    /**
     * Returns the weight vectors.
     */
    public double[][] getWeights() {
        double[][] copy = new double[numClasses][];
        for (int i = 0; i < numClasses; i++) {
            copy[i] = Arrays.copyOf(weights[i], numFeatures);
        }
        return copy;
    }
    
    /**
     * Returns the bias terms.
     */
    public double[] getBias() {
        return Arrays.copyOf(bias, bias.length);
    }
    
    /**
     * Returns whether the model has been trained.
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * Builder class for SVC.
     */
    public static class Builder {
        private double C = 1.0;
        private int maxIter = 1000;
        private double tol = 1e-3;
        private double learningRate = 0.01;
        
        public Builder C(double C) {
            if (C <= 0.0) {
                throw new IllegalArgumentException("C must be positive");
            }
            this.C = C;
            return this;
        }
        
        public Builder maxIter(int maxIter) {
            if (maxIter <= 0) {
                throw new IllegalArgumentException("maxIter must be positive");
            }
            this.maxIter = maxIter;
            return this;
        }
        
        public Builder tol(double tol) {
            if (tol <= 0.0) {
                throw new IllegalArgumentException("tol must be positive");
            }
            this.tol = tol;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0.0) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        public SVC build() {
            return new SVC(C, maxIter, tol, learningRate);
        }
    }
}
