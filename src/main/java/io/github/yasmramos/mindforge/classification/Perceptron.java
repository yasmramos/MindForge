package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * Perceptron classifier.
 * 
 * <p>The Perceptron is a simple linear classifier that learns a hyperplane
 * to separate classes. It works well for linearly separable data.</p>
 * 
 * <p>This implementation supports:
 * <ul>
 *   <li>Binary classification</li>
 *   <li>Multi-class classification (One-vs-All)</li>
 *   <li>Configurable learning rate and iterations</li>
 *   <li>Early stopping when no errors</li>
 * </ul>
 * </p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * Perceptron perceptron = new Perceptron(0.1, 100, 42);
 * perceptron.train(X_train, y_train);
 * int[] predictions = perceptron.predict(X_test);
 * }</pre>
 * 
 * @author Matrix Agent
 * @version 1.0
 */
public class Perceptron implements Classifier<double[]>, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final double learningRate;
    private final int maxIterations;
    private final int randomState;
    private final boolean shuffle;
    
    private double[][] weights;  // [numClasses][numFeatures + 1] (including bias)
    private int[] classes;
    private int numFeatures;
    private boolean isTrained;
    private Random random;
    
    /**
     * Creates a Perceptron with default settings.
     * Uses learning rate 0.01, 1000 max iterations.
     */
    public Perceptron() {
        this(0.01, 1000, -1, true);
    }
    
    /**
     * Creates a Perceptron with specified learning rate.
     * 
     * @param learningRate the learning rate (eta)
     */
    public Perceptron(double learningRate) {
        this(learningRate, 1000, -1, true);
    }
    
    /**
     * Creates a Perceptron with specified parameters.
     * 
     * @param learningRate the learning rate
     * @param maxIterations maximum number of passes over training data
     * @param randomState random seed (-1 for random)
     */
    public Perceptron(double learningRate, int maxIterations, int randomState) {
        this(learningRate, maxIterations, randomState, true);
    }
    
    /**
     * Creates a Perceptron with full configuration.
     * 
     * @param learningRate the learning rate (must be positive)
     * @param maxIterations maximum iterations (must be positive)
     * @param randomState random seed (-1 for random)
     * @param shuffle whether to shuffle data each epoch
     * @throws IllegalArgumentException if parameters are invalid
     */
    public Perceptron(double learningRate, int maxIterations, int randomState, boolean shuffle) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive, got: " + learningRate);
        }
        if (maxIterations < 1) {
            throw new IllegalArgumentException("Max iterations must be at least 1, got: " + maxIterations);
        }
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.randomState = randomState;
        this.shuffle = shuffle;
        this.isTrained = false;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        validateInput(X, y);
        
        this.numFeatures = X[0].length;
        this.random = randomState >= 0 ? new Random(randomState) : new Random();
        
        // Find unique classes
        Set<Integer> uniqueClasses = new TreeSet<>();
        for (int label : y) {
            uniqueClasses.add(label);
        }
        this.classes = uniqueClasses.stream().mapToInt(Integer::intValue).toArray();
        
        if (classes.length < 2) {
            throw new IllegalArgumentException("Need at least 2 classes for classification");
        }
        
        if (classes.length == 2) {
            // Binary classification
            trainBinary(X, y);
        } else {
            // Multi-class: One-vs-All
            trainMulticlass(X, y);
        }
        
        this.isTrained = true;
    }
    
    /**
     * Trains binary perceptron.
     */
    private void trainBinary(double[][] X, int[] y) {
        int n = X.length;
        weights = new double[1][numFeatures + 1]; // +1 for bias
        
        // Convert labels to +1/-1
        int[] binaryY = new int[n];
        for (int i = 0; i < n; i++) {
            binaryY[i] = y[i] == classes[1] ? 1 : -1;
        }
        
        // Training
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            if (shuffle) {
                shuffleArray(indices);
            }
            
            int errors = 0;
            for (int idx : indices) {
                double activation = computeActivation(X[idx], 0);
                int predicted = activation >= 0 ? 1 : -1;
                
                if (predicted != binaryY[idx]) {
                    errors++;
                    // Update weights
                    double update = learningRate * binaryY[idx];
                    for (int j = 0; j < numFeatures; j++) {
                        weights[0][j] += update * X[idx][j];
                    }
                    weights[0][numFeatures] += update; // bias
                }
            }
            
            // Early stopping if no errors
            if (errors == 0) {
                break;
            }
        }
    }
    
    /**
     * Trains multi-class perceptron using One-vs-All.
     */
    private void trainMulticlass(double[][] X, int[] y) {
        int n = X.length;
        int numClasses = classes.length;
        weights = new double[numClasses][numFeatures + 1];
        
        // Train one classifier per class
        for (int c = 0; c < numClasses; c++) {
            int targetClass = classes[c];
            
            // Convert to binary: target class = +1, rest = -1
            int[] binaryY = new int[n];
            for (int i = 0; i < n; i++) {
                binaryY[i] = y[i] == targetClass ? 1 : -1;
            }
            
            // Training for this class
            int[] indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            
            for (int iter = 0; iter < maxIterations; iter++) {
                if (shuffle) {
                    shuffleArray(indices);
                }
                
                int errors = 0;
                for (int idx : indices) {
                    double activation = computeActivation(X[idx], c);
                    int predicted = activation >= 0 ? 1 : -1;
                    
                    if (predicted != binaryY[idx]) {
                        errors++;
                        double update = learningRate * binaryY[idx];
                        for (int j = 0; j < numFeatures; j++) {
                            weights[c][j] += update * X[idx][j];
                        }
                        weights[c][numFeatures] += update;
                    }
                }
                
                if (errors == 0) {
                    break;
                }
            }
        }
    }
    
    /**
     * Computes the activation (weighted sum) for a sample.
     */
    private double computeActivation(double[] x, int classIndex) {
        double sum = weights[classIndex][numFeatures]; // bias
        for (int j = 0; j < numFeatures; j++) {
            sum += weights[classIndex][j] * x[j];
        }
        return sum;
    }
    
    @Override
    public int predict(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("Perceptron must be trained before prediction");
        }
        validatePredictInput(x);
        
        if (classes.length == 2) {
            double activation = computeActivation(x, 0);
            return activation >= 0 ? classes[1] : classes[0];
        } else {
            // Multi-class: return class with highest activation
            int bestClass = classes[0];
            double bestActivation = Double.NEGATIVE_INFINITY;
            
            for (int c = 0; c < classes.length; c++) {
                double activation = computeActivation(x, c);
                if (activation > bestActivation) {
                    bestActivation = activation;
                    bestClass = classes[c];
                }
            }
            return bestClass;
        }
    }
    
    /**
     * Predicts class labels for multiple samples.
     * 
     * @param X feature matrix
     * @return predicted labels
     */
    public int[] predict(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Returns decision function values for each class.
     * 
     * @param X feature matrix
     * @return decision values [n_samples][n_classes]
     */
    public double[][] decisionFunction(double[][] X) {
        if (!isTrained) {
            throw new IllegalStateException("Perceptron must be trained first");
        }
        
        int n = X.length;
        int numClasses = classes.length == 2 ? 1 : classes.length;
        double[][] decisions = new double[n][numClasses];
        
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < numClasses; c++) {
                decisions[i][c] = computeActivation(X[i], c);
            }
        }
        
        return decisions;
    }
    
    @Override
    public int getNumClasses() {
        return isTrained ? classes.length : 0;
    }
    
    /**
     * Gets the learned weights.
     * 
     * @return weight matrix [numClasses][numFeatures + 1]
     */
    public double[][] getWeights() {
        if (!isTrained) {
            throw new IllegalStateException("Perceptron must be trained first");
        }
        double[][] copy = new double[weights.length][];
        for (int i = 0; i < weights.length; i++) {
            copy[i] = weights[i].clone();
        }
        return copy;
    }
    
    /**
     * Gets the class labels.
     * 
     * @return array of class labels
     */
    public int[] getClasses() {
        if (!isTrained) {
            throw new IllegalStateException("Perceptron must be trained first");
        }
        return classes.clone();
    }
    
    /**
     * Checks if the classifier is trained.
     * 
     * @return true if trained
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * Gets the learning rate.
     * 
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Gets the maximum iterations.
     * 
     * @return max iterations
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    private void validateInput(double[][] X, int[] y) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Labels cannot be null or empty");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException(
                String.format("X and y have different lengths: %d vs %d", X.length, y.length));
        }
    }
    
    private void validatePredictInput(double[] x) {
        if (x == null) {
            throw new IllegalArgumentException("Input cannot be null");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length));
        }
    }
    
    private void shuffleArray(int[] array) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
}
