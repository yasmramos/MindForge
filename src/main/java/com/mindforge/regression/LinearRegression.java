package com.mindforge.regression;

/**
 * Simple Linear Regression using Ordinary Least Squares.
 * Fits a linear model to minimize the sum of squared residuals.
 */
public class LinearRegression implements Regressor<double[]> {
    
    private double[] weights;
    private double bias;
    private boolean fitted;
    
    /**
     * Creates a new Linear Regression model.
     */
    public LinearRegression() {
        this.fitted = false;
    }
    
    @Override
    public void train(double[][] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("Data and target must have the same length");
        }
        if (x.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        int n = x.length;
        int features = x[0].length;
        
        // Initialize weights and bias
        weights = new double[features];
        bias = 0.0;
        
        // Using Normal Equation: w = (X^T X)^-1 X^T y
        // For simplicity, using gradient descent approach
        double learningRate = 0.01;
        int iterations = 1000;
        
        for (int iter = 0; iter < iterations; iter++) {
            // Calculate gradients
            double[] gradWeights = new double[features];
            double gradBias = 0.0;
            
            for (int i = 0; i < n; i++) {
                double prediction = predictInternal(x[i]);
                double error = prediction - y[i];
                
                gradBias += error;
                for (int j = 0; j < features; j++) {
                    gradWeights[j] += error * x[i][j];
                }
            }
            
            // Update weights and bias
            bias -= learningRate * gradBias / n;
            for (int j = 0; j < features; j++) {
                weights[j] -= learningRate * gradWeights[j] / n;
            }
        }
        
        fitted = true;
    }
    
    /**
     * Internal prediction method used during training.
     * Does not check if model is fitted.
     * 
     * @param x input features
     * @return predicted value
     */
    private double predictInternal(double[] x) {
        double prediction = bias;
        for (int i = 0; i < x.length; i++) {
            prediction += weights[i] * x[i];
        }
        return prediction;
    }
    
    @Override
    public double predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != weights.length) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        
        double prediction = bias;
        for (int i = 0; i < x.length; i++) {
            prediction += weights[i] * x[i];
        }
        return prediction;
    }
    
    /**
     * Returns the learned weights.
     * 
     * @return array of weights
     */
    public double[] getWeights() {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained first");
        }
        return weights.clone();
    }
    
    /**
     * Returns the learned bias term.
     * 
     * @return bias value
     */
    public double getBias() {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained first");
        }
        return bias;
    }
    
    /**
     * Checks if the model has been trained.
     * 
     * @return true if model is fitted
     */
    public boolean isFitted() {
        return fitted;
    }
}
