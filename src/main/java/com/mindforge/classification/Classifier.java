package com.mindforge.classification;

/**
 * Interface for classification algorithms.
 * A classifier predicts categorical labels for input data.
 * 
 * @param <T> the type of input data
 */
public interface Classifier<T> {
    
    /**
     * Trains the classifier with the given training data.
     * 
     * @param x training data features
     * @param y training data labels
     */
    void train(T[] x, int[] y);
    
    /**
     * Predicts the label for a single input.
     * 
     * @param x input data
     * @return predicted label
     */
    int predict(T x);
    
    /**
     * Predicts labels for multiple inputs.
     * 
     * @param x array of input data
     * @return array of predicted labels
     */
    default int[] predict(T[] x) {
        int[] predictions = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            predictions[i] = predict(x[i]);
        }
        return predictions;
    }
    
    /**
     * Returns the number of classes this classifier can predict.
     * 
     * @return number of classes
     */
    int getNumClasses();
}
