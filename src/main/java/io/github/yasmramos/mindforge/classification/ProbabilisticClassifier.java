package io.github.yasmramos.mindforge.classification;

/**
 * Interface for classifiers that can provide probability estimates.
 * Extends the base Classifier interface with probabilistic prediction capabilities.
 * 
 * @param <T> the type of input data
 * @author MindForge
 */
public interface ProbabilisticClassifier<T> extends Classifier<T> {
    
    /**
     * Predicts class probabilities for a single input.
     * 
     * @param x input data
     * @return array of probabilities for each class
     */
    double[] predictProba(T x);
    
    /**
     * Predicts class probabilities for multiple inputs.
     * 
     * @param x array of input data
     * @return 2D array of probabilities (samples x classes)
     */
    default double[][] predictProba(T[] x) {
        double[][] probas = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            probas[i] = predictProba(x[i]);
        }
        return probas;
    }
    
    /**
     * Calculates the accuracy score on the given test data.
     * 
     * @param X test features
     * @param y true labels
     * @return accuracy score between 0.0 and 1.0
     */
    default double score(T[] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        return (double) correct / y.length;
    }
    
    /**
     * Checks if the classifier has been trained.
     * 
     * @return true if trained, false otherwise
     */
    boolean isTrained();
}
