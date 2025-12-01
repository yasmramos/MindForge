package com.mindforge.classification;

import java.util.*;

/**
 * Bernoulli Naive Bayes classifier.
 * 
 * Suitable for binary/boolean features (0 or 1).
 * Like MultinomialNB but penalizes non-occurrence of features.
 * 
 * Models each feature as a Bernoulli distribution (binary outcome).
 * P(feature_i = 1 | class) is estimated from training data.
 * P(feature_i = 0 | class) = 1 - P(feature_i = 1 | class)
 * 
 * Commonly used for text classification with binary word occurrence features.
 * 
 * Example usage:
 * <pre>
 * BernoulliNaiveBayes bnb = new BernoulliNaiveBayes(1.0); // alpha=1.0 for smoothing
 * bnb.train(X_train, y_train);
 * int[] predictions = bnb.predict(X_test);
 * </pre>
 */
public class BernoulliNaiveBayes implements Classifier<double[]> {
    
    private double alpha; // Smoothing parameter
    private double binarize; // Threshold for converting features to binary
    private boolean fitPrior;
    
    private int numClasses;
    private int numFeatures;
    private int[] classes;
    private double[] classPriors;
    private double[][] featureProbs; // [class][feature] P(feature=1|class)
    
    private boolean isTrained = false;
    
    /**
     * Creates a new Bernoulli Naive Bayes classifier with default settings.
     * Uses Laplace smoothing (alpha=1.0), no binarization, and fitted priors.
     */
    public BernoulliNaiveBayes() {
        this(1.0, -1.0, true);
    }
    
    /**
     * Creates a new Bernoulli Naive Bayes classifier with custom smoothing.
     * 
     * @param alpha smoothing parameter (typically 1.0 for Laplace smoothing)
     */
    public BernoulliNaiveBayes(double alpha) {
        this(alpha, -1.0, true);
    }
    
    /**
     * Creates a new Bernoulli Naive Bayes classifier with full customization.
     * 
     * @param alpha smoothing parameter
     * @param binarize threshold for binarization (-1.0 to disable, otherwise features > threshold become 1)
     * @param fitPrior whether to learn class prior probabilities (if false, uniform priors assumed)
     */
    public BernoulliNaiveBayes(double alpha, double binarize, boolean fitPrior) {
        if (alpha < 0.0) {
            throw new IllegalArgumentException("Alpha must be non-negative");
        }
        this.alpha = alpha;
        this.binarize = binarize;
        this.fitPrior = fitPrior;
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
        featureProbs = new double[numClasses][numFeatures];
        int[] classCounts = new int[numClasses];
        double[][] featureCounts = new double[numClasses][numFeatures];
        
        // Binarize data if needed and count features
        for (int i = 0; i < n; i++) {
            int classIdx = classIndexMap.get(y[i]);
            classCounts[classIdx]++;
            
            for (int f = 0; f < numFeatures; f++) {
                double value = X[i][f];
                
                // Binarize if threshold is set
                if (binarize >= 0.0) {
                    value = value > binarize ? 1.0 : 0.0;
                }
                
                // Count if feature is present (1)
                if (value > 0.0) {
                    featureCounts[classIdx][f]++;
                }
            }
        }
        
        // Calculate priors and feature probabilities
        for (int c = 0; c < numClasses; c++) {
            if (classCounts[c] == 0) {
                continue;
            }
            
            // Class prior
            if (fitPrior) {
                classPriors[c] = (double) classCounts[c] / n;
            } else {
                classPriors[c] = 1.0 / numClasses; // Uniform prior
            }
            
            // Feature probabilities with smoothing
            for (int f = 0; f < numFeatures; f++) {
                // P(feature=1|class) with Laplace smoothing
                double smoothedCount = featureCounts[c][f] + alpha;
                double smoothedTotal = classCounts[c] + 2 * alpha; // +2*alpha for binary (0,1)
                featureProbs[c][f] = smoothedCount / smoothedTotal;
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
                double value = x[f];
                
                // Binarize if threshold is set
                if (binarize >= 0.0) {
                    value = value > binarize ? 1.0 : 0.0;
                }
                
                // P(feature|class)
                if (value > 0.0) {
                    // Feature is present
                    logProb += Math.log(featureProbs[c][f]);
                } else {
                    // Feature is absent (this is what makes Bernoulli different from Multinomial)
                    logProb += Math.log(1.0 - featureProbs[c][f]);
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
                double value = x[f];
                
                // Binarize if threshold is set
                if (binarize >= 0.0) {
                    value = value > binarize ? 1.0 : 0.0;
                }
                
                if (value > 0.0) {
                    logProbs[c] += Math.log(featureProbs[c][f]);
                } else {
                    logProbs[c] += Math.log(1.0 - featureProbs[c][f]);
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
     * Returns the feature probabilities.
     * 
     * @return 2D array of P(feature=1|class) probabilities [class][feature]
     */
    public double[][] getFeatureProbs() {
        double[][] copy = new double[numClasses][];
        for (int i = 0; i < numClasses; i++) {
            copy[i] = Arrays.copyOf(featureProbs[i], numFeatures);
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
     * Returns the binarization threshold.
     * 
     * @return binarize threshold (-1.0 if disabled)
     */
    public double getBinarize() {
        return binarize;
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
