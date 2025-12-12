package io.github.yasmramos.mindforge.classification;

import java.util.*;

/**
 * Multinomial Naive Bayes classifier.
 * 
 * Suitable for classification with discrete features (e.g., word counts for text classification).
 * The features represent frequencies or counts.
 * 
 * Uses the multinomial distribution to model feature probabilities:
 * P(feature_i = count | class) = (count + alpha) / (sum_counts + alpha * n_features)
 * 
 * Where alpha is the smoothing parameter (Laplace/Lidstone smoothing).
 * 
 * Example usage:
 * <pre>
 * MultinomialNaiveBayes mnb = new MultinomialNaiveBayes(1.0); // alpha=1.0 for Laplace smoothing
 * mnb.train(X_train, y_train);
 * int[] predictions = mnb.predict(X_test);
 * </pre>
 */
public class MultinomialNaiveBayes implements Classifier<double[]> {
    
    private double alpha; // Smoothing parameter
    private int numClasses;
    private int numFeatures;
    private int[] classes;
    private double[] classPriors;
    private double[][] featureLogProbs; // [class][feature] log probabilities
    
    private boolean isTrained = false;
    
    /**
     * Creates a new Multinomial Naive Bayes classifier with Laplace smoothing (alpha=1.0).
     */
    public MultinomialNaiveBayes() {
        this(1.0);
    }
    
    /**
     * Creates a new Multinomial Naive Bayes classifier with custom smoothing.
     * 
     * @param alpha smoothing parameter (typically 0.0 to 1.0)
     *              - alpha=1.0: Laplace smoothing
     *              - alpha=0.0: No smoothing
     */
    public MultinomialNaiveBayes(double alpha) {
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
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
        featureLogProbs = new double[numClasses][numFeatures];
        double[][] featureCounts = new double[numClasses][numFeatures];
        double[] classCounts = new double[numClasses];
        
        // Count features for each class
        for (int i = 0; i < n; i++) {
            int classIdx = classIndexMap.get(y[i]);
            classCounts[classIdx]++;
            
            for (int f = 0; f < numFeatures; f++) {
                if (X[i][f] < 0.0) {
                    throw new IllegalArgumentException(
                        "Multinomial Naive Bayes requires non-negative feature values"
                    );
                }
                featureCounts[classIdx][f] += X[i][f];
            }
        }
        
        // Calculate priors and feature probabilities
        for (int c = 0; c < numClasses; c++) {
            if (classCounts[c] == 0) {
                continue;
            }
            
            // Class prior
            classPriors[c] = classCounts[c] / n;
            
            // Total count for this class (sum of all feature counts)
            double totalCount = 0.0;
            for (int f = 0; f < numFeatures; f++) {
                totalCount += featureCounts[c][f];
            }
            
            // Feature log probabilities with smoothing
            for (int f = 0; f < numFeatures; f++) {
                double smoothedCount = featureCounts[c][f] + alpha;
                double smoothedTotal = totalCount + alpha * numFeatures;
                featureLogProbs[c][f] = Math.log(smoothedCount / smoothedTotal);
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
            
            // Add weighted feature log probabilities
            for (int f = 0; f < numFeatures; f++) {
                if (x[f] < 0.0) {
                    throw new IllegalArgumentException(
                        "Multinomial Naive Bayes requires non-negative feature values"
                    );
                }
                if (x[f] > 0.0) {
                    logProb += x[f] * featureLogProbs[c][f];
                }
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
                if (x[f] < 0.0) {
                    throw new IllegalArgumentException(
                        "Multinomial Naive Bayes requires non-negative feature values"
                    );
                }
                if (x[f] > 0.0) {
                    logProbs[c] += x[f] * featureLogProbs[c][f];
                }
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
     * Returns the feature log probabilities.
     * 
     * @return 2D array of log probabilities [class][feature]
     */
    public double[][] getFeatureLogProbs() {
        double[][] copy = new double[numClasses][];
        for (int i = 0; i < numClasses; i++) {
            copy[i] = Arrays.copyOf(featureLogProbs[i], numFeatures);
        }
        return copy;
    }
    
    /**
     * Returns the smoothing parameter.
     * 
     * @return alpha value
     */
    public double getAlpha() {
        return alpha;
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
