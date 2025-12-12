package io.github.yasmramos.mindforge.classification;

import java.util.*;

/**
 * Gaussian Naive Bayes classifier.
 * 
 * Assumes that features follow a Gaussian (normal) distribution within each class.
 * This is suitable for continuous features.
 * 
 * The algorithm calculates the mean and variance of each feature for each class
 * during training, then uses these statistics to compute class probabilities
 * during prediction using Bayes' theorem.
 * 
 * P(class | features) ∝ P(class) * ∏ P(feature_i | class)
 * 
 * Where P(feature_i | class) follows a Gaussian distribution:
 * P(x | class) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
 * 
 * Example usage:
 * <pre>
 * GaussianNaiveBayes gnb = new GaussianNaiveBayes();
 * gnb.train(X_train, y_train);
 * int[] predictions = gnb.predict(X_test);
 * double[][] probabilities = gnb.predictProba(X_test);
 * </pre>
 */
public class GaussianNaiveBayes implements Classifier<double[]> {
    
    private int numClasses;
    private int numFeatures;
    private int[] classes;
    private double[] classPriors;
    private double[][] means;      // [class][feature]
    private double[][] variances;  // [class][feature]
    private double epsilon = 1e-9; // Small value to prevent division by zero
    
    private boolean isTrained = false;
    
    /**
     * Creates a new Gaussian Naive Bayes classifier.
     */
    public GaussianNaiveBayes() {
        // Default constructor
    }
    
    /**
     * Creates a new Gaussian Naive Bayes classifier with custom epsilon.
     * 
     * @param epsilon small value added to variance to prevent division by zero
     */
    public GaussianNaiveBayes(double epsilon) {
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive");
        }
        this.epsilon = epsilon;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same number of samples");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        int n = X.length;
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
        
        // Create class index mapping
        Map<Integer, Integer> classIndexMap = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            classIndexMap.put(classes[i], i);
        }
        
        // Initialize storage
        classPriors = new double[numClasses];
        means = new double[numClasses][numFeatures];
        variances = new double[numClasses][numFeatures];
        
        // Group samples by class
        List<List<double[]>> classSamples = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            classSamples.add(new ArrayList<>());
        }
        
        for (int i = 0; i < n; i++) {
            int classIdx = classIndexMap.get(y[i]);
            classSamples.get(classIdx).add(X[i]);
        }
        
        // Calculate statistics for each class
        for (int c = 0; c < numClasses; c++) {
            List<double[]> samples = classSamples.get(c);
            int classSize = samples.size();
            
            if (classSize == 0) {
                continue;
            }
            
            // Class prior probability
            classPriors[c] = (double) classSize / n;
            
            // Calculate mean for each feature
            for (int f = 0; f < numFeatures; f++) {
                double sum = 0.0;
                for (double[] sample : samples) {
                    sum += sample[f];
                }
                means[c][f] = sum / classSize;
            }
            
            // Calculate variance for each feature
            for (int f = 0; f < numFeatures; f++) {
                double sum = 0.0;
                for (double[] sample : samples) {
                    double diff = sample[f] - means[c][f];
                    sum += diff * diff;
                }
                variances[c][f] = sum / classSize + epsilon;
            }
        }
        
        isTrained = true;
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
        
        double maxLogProb = Double.NEGATIVE_INFINITY;
        int predictedClass = classes[0];
        
        for (int c = 0; c < numClasses; c++) {
            double logProb = Math.log(classPriors[c]);
            
            // Calculate log probability for each feature
            for (int f = 0; f < numFeatures; f++) {
                logProb += logGaussianProbability(x[f], means[c][f], variances[c][f]);
            }
            
            if (logProb > maxLogProb) {
                maxLogProb = logProb;
                predictedClass = classes[c];
            }
        }
        
        return predictedClass;
    }
    
    /**
     * Predicts class labels for multiple samples.
     * 
     * @param X array of feature vectors
     * @return array of predicted class labels
     */
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Predicts class probabilities for a single sample.
     * 
     * @param x feature vector
     * @return array of class probabilities (one per class)
     */
    public double[] predictProba(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", numFeatures, x.length)
            );
        }
        
        double[] logProbs = new double[numClasses];
        double maxLogProb = Double.NEGATIVE_INFINITY;
        
        // Calculate log probabilities
        for (int c = 0; c < numClasses; c++) {
            logProbs[c] = Math.log(classPriors[c]);
            
            for (int f = 0; f < numFeatures; f++) {
                logProbs[c] += logGaussianProbability(x[f], means[c][f], variances[c][f]);
            }
            
            if (logProbs[c] > maxLogProb) {
                maxLogProb = logProbs[c];
            }
        }
        
        // Convert to probabilities using log-sum-exp trick
        double[] probs = new double[numClasses];
        double sumExp = 0.0;
        
        for (int c = 0; c < numClasses; c++) {
            probs[c] = Math.exp(logProbs[c] - maxLogProb);
            sumExp += probs[c];
        }
        
        // Normalize
        for (int c = 0; c < numClasses; c++) {
            probs[c] /= sumExp;
        }
        
        return probs;
    }
    
    /**
     * Predicts class probabilities for multiple samples.
     * 
     * @param X array of feature vectors
     * @return 2D array of class probabilities [sample][class]
     */
    public double[][] predictProba(double[][] X) {
        double[][] probabilities = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProba(X[i]);
        }
        return probabilities;
    }
    
    /**
     * Calculates the log of Gaussian probability density function.
     * 
     * @param x value
     * @param mean mean of the distribution
     * @param variance variance of the distribution
     * @return log probability
     */
    private double logGaussianProbability(double x, double mean, double variance) {
        double diff = x - mean;
        return -0.5 * Math.log(2 * Math.PI * variance) - (diff * diff) / (2 * variance);
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Returns the class labels.
     * 
     * @return array of class labels
     */
    public int[] getClasses() {
        return Arrays.copyOf(classes, classes.length);
    }
    
    /**
     * Returns the class prior probabilities.
     * 
     * @return array of prior probabilities
     */
    public double[] getClassPriors() {
        return Arrays.copyOf(classPriors, classPriors.length);
    }
    
    /**
     * Returns the means for each class and feature.
     * 
     * @return 2D array of means [class][feature]
     */
    public double[][] getMeans() {
        double[][] copy = new double[numClasses][];
        for (int i = 0; i < numClasses; i++) {
            copy[i] = Arrays.copyOf(means[i], numFeatures);
        }
        return copy;
    }
    
    /**
     * Returns the variances for each class and feature.
     * 
     * @return 2D array of variances [class][feature]
     */
    public double[][] getVariances() {
        double[][] copy = new double[numClasses][];
        for (int i = 0; i < numClasses; i++) {
            copy[i] = Arrays.copyOf(variances[i], numFeatures);
        }
        return copy;
    }
    
    /**
     * Returns whether the model has been trained.
     * 
     * @return true if trained, false otherwise
     */
    public boolean isTrained() {
        return isTrained;
    }
}
