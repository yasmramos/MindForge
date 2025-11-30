package com.mindforge.regression;

/**
 * Interface for regression algorithms.
 * A regressor predicts continuous values for input data.
 * 
 * @param <T> the type of input data
 */
public interface Regressor<T> {
    
    /**
     * Trains the regressor with the given training data.
     * 
     * @param x training data features
     * @param y training data target values
     */
    void train(T[] x, double[] y);
    
    /**
     * Predicts the value for a single input.
     * 
     * @param x input data
     * @return predicted value
     */
    double predict(T x);
    
    /**
     * Predicts values for multiple inputs.
     * 
     * @param x array of input data
     * @return array of predicted values
     */
    default double[] predict(T[] x) {
        double[] predictions = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            predictions[i] = predict(x[i]);
        }
        return predictions;
    }
}
