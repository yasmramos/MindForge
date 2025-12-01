package com.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * An AdaBoost classifier.
 * 
 * <p>An AdaBoost classifier is a meta-estimator that begins by fitting a classifier
 * on the original dataset and then fits additional copies of the classifier on the
 * same dataset but where the weights of incorrectly classified instances are adjusted
 * such that subsequent classifiers focus more on difficult cases.</p>
 * 
 * <p>This implementation uses decision stumps (1-level decision trees) as weak learners.</p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * AdaBoostClassifier ada = new AdaBoostClassifier(50);
 * ada.fit(X_train, y_train);
 * int[] predictions = ada.predict(X_test);
 * }</pre>
 * 
 * @author Matrix Agent
 * @version 1.0
 */
public class AdaBoostClassifier implements Classifier<double[]>, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final int nEstimators;
    private final double learningRate;
    private final int randomState;
    
    private List<DecisionStump> stumps;
    private List<Double> stumpWeights;
    private int[] classes;
    private int nFeatures;
    private boolean isFitted;
    private Random random;
    
    /**
     * Creates an AdaBoostClassifier with default settings.
     * Uses 50 estimators and learning rate of 1.0.
     */
    public AdaBoostClassifier() {
        this(50, 1.0, -1);
    }
    
    /**
     * Creates an AdaBoostClassifier with specified number of estimators.
     * 
     * @param nEstimators The maximum number of estimators
     */
    public AdaBoostClassifier(int nEstimators) {
        this(nEstimators, 1.0, -1);
    }
    
    /**
     * Creates an AdaBoostClassifier with full configuration.
     * 
     * @param nEstimators The maximum number of estimators
     * @param learningRate Weight applied to each classifier at each boosting iteration
     * @param randomState Random seed for reproducibility (-1 for random)
     * @throws IllegalArgumentException if nEstimators < 1 or learningRate <= 0
     */
    public AdaBoostClassifier(int nEstimators, double learningRate, int randomState) {
        if (nEstimators < 1) {
            throw new IllegalArgumentException("nEstimators must be at least 1, got: " + nEstimators);
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got: " + learningRate);
        }
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.randomState = randomState;
        this.isFitted = false;
    }
    
    /**
     * Trains the classifier with the given training data.
     * 
     * @param x training data features
     * @param y training data labels
     */
    @Override
    public void train(double[][] x, int[] y) {
        fit(x, y);
    }
    
    /**
     * Fit the AdaBoost classifier.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @param y Target labels
     */
    public void fit(double[][] X, int[] y) {
        validateInput(X, y);
        
        this.nFeatures = X[0].length;
        this.random = randomState >= 0 ? new Random(randomState) : new Random();
        
        // Find unique classes
        Set<Integer> uniqueClasses = new TreeSet<>();
        for (int label : y) {
            uniqueClasses.add(label);
        }
        this.classes = uniqueClasses.stream().mapToInt(Integer::intValue).toArray();
        
        if (classes.length != 2) {
            throw new IllegalArgumentException("AdaBoost currently supports binary classification only. Found " + classes.length + " classes.");
        }
        
        int nSamples = X.length;
        double[] sampleWeights = new double[nSamples];
        Arrays.fill(sampleWeights, 1.0 / nSamples);
        
        this.stumps = new ArrayList<>();
        this.stumpWeights = new ArrayList<>();
        
        for (int t = 0; t < nEstimators; t++) {
            // Fit a decision stump
            DecisionStump stump = new DecisionStump();
            stump.fit(X, y, sampleWeights);
            
            // Get predictions
            int[] predictions = stump.predict(X);
            
            // Calculate weighted error
            double error = 0.0;
            for (int i = 0; i < nSamples; i++) {
                if (predictions[i] != y[i]) {
                    error += sampleWeights[i];
                }
            }
            
            // Avoid division by zero and log(0)
            error = Math.max(error, 1e-10);
            error = Math.min(error, 1.0 - 1e-10);
            
            // If error is too high, stop
            if (error >= 0.5) {
                if (stumps.isEmpty()) {
                    // Keep at least one stump
                    stumps.add(stump);
                    stumpWeights.add(1.0);
                }
                break;
            }
            
            // Calculate stump weight (alpha)
            double alpha = learningRate * 0.5 * Math.log((1.0 - error) / error);
            
            stumps.add(stump);
            stumpWeights.add(alpha);
            
            // Update sample weights
            double sumWeights = 0.0;
            for (int i = 0; i < nSamples; i++) {
                double indicator = predictions[i] == y[i] ? -1.0 : 1.0;
                sampleWeights[i] *= Math.exp(alpha * indicator);
                sumWeights += sampleWeights[i];
            }
            
            // Normalize weights
            for (int i = 0; i < nSamples; i++) {
                sampleWeights[i] /= sumWeights;
            }
        }
        
        this.isFitted = true;
    }
    
    /**
     * Predicts the label for a single input.
     * 
     * @param x input data
     * @return predicted label
     */
    @Override
    public int predict(double[] x) {
        return predict(new double[][]{x})[0];
    }
    
    /**
     * Returns the number of classes this classifier can predict.
     * 
     * @return number of classes (2 for binary classification)
     */
    @Override
    public int getNumClasses() {
        if (!isFitted) {
            return 2; // AdaBoost is binary
        }
        return classes.length;
    }
    
    /**
     * Predict class labels for samples in X.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return Predicted class labels
     */
    public int[] predict(double[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted before predict");
        }
        validatePredictInput(X);
        
        int nSamples = X.length;
        int[] predictions = new int[nSamples];
        double[] scores = predictScore(X);
        
        for (int i = 0; i < nSamples; i++) {
            predictions[i] = scores[i] >= 0 ? classes[1] : classes[0];
        }
        
        return predictions;
    }
    
    /**
     * Predict class probabilities for X.
     * 
     * @param X Input data of shape [n_samples, n_features]
     * @return Class probabilities of shape [n_samples, n_classes]
     */
    public double[][] predictProba(double[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted before predict");
        }
        validatePredictInput(X);
        
        int nSamples = X.length;
        double[][] proba = new double[nSamples][2];
        double[] scores = predictScore(X);
        
        for (int i = 0; i < nSamples; i++) {
            // Convert score to probability using sigmoid
            double prob = 1.0 / (1.0 + Math.exp(-2.0 * scores[i]));
            proba[i][0] = 1.0 - prob;
            proba[i][1] = prob;
        }
        
        return proba;
    }
    
    /**
     * Get the weighted sum of stump predictions.
     */
    private double[] predictScore(double[][] X) {
        int nSamples = X.length;
        double[] scores = new double[nSamples];
        
        for (int t = 0; t < stumps.size(); t++) {
            int[] preds = stumps.get(t).predict(X);
            double alpha = stumpWeights.get(t);
            
            for (int i = 0; i < nSamples; i++) {
                // Convert to +1/-1
                double sign = preds[i] == classes[1] ? 1.0 : -1.0;
                scores[i] += alpha * sign;
            }
        }
        
        return scores;
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
    
    private void validatePredictInput(double[][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Input data cannot be null or empty");
        }
        if (X[0].length != nFeatures) {
            throw new IllegalArgumentException(
                String.format("X has %d features, but AdaBoostClassifier is expecting %d features",
                    X[0].length, nFeatures));
        }
    }
    
    /**
     * Get the number of estimators actually used.
     * 
     * @return number of estimators
     */
    public int getActualNEstimators() {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted first");
        }
        return stumps.size();
    }
    
    /**
     * Get the weight of each estimator.
     * 
     * @return array of estimator weights
     */
    public double[] getEstimatorWeights() {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted first");
        }
        return stumpWeights.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Get feature importances.
     * 
     * @return array of feature importances
     */
    public double[] getFeatureImportances() {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted first");
        }
        
        double[] importances = new double[nFeatures];
        double totalWeight = stumpWeights.stream().mapToDouble(Double::doubleValue).sum();
        
        for (int t = 0; t < stumps.size(); t++) {
            int feature = stumps.get(t).getFeatureIndex();
            importances[feature] += stumpWeights.get(t);
        }
        
        // Normalize
        if (totalWeight > 0) {
            for (int i = 0; i < nFeatures; i++) {
                importances[i] /= totalWeight;
            }
        }
        
        return importances;
    }
    
    /**
     * Get the classes.
     * 
     * @return array of class labels
     */
    public int[] getClasses() {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted first");
        }
        return classes.clone();
    }
    
    /**
     * Check if the classifier has been fitted.
     * 
     * @return true if fitted
     */
    public boolean isFitted() {
        return isFitted;
    }
    
    /**
     * Get the configured number of estimators.
     * 
     * @return configured number of estimators
     */
    public int getNEstimators() {
        return nEstimators;
    }
    
    /**
     * Get the learning rate.
     * 
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * Decision stump - a decision tree with depth 1.
     */
    private static class DecisionStump implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private int featureIndex;
        private double threshold;
        private int leftClass;
        private int rightClass;
        
        public void fit(double[][] X, int[] y, double[] weights) {
            int nSamples = X.length;
            int nFeatures = X[0].length;
            
            double bestGain = Double.NEGATIVE_INFINITY;
            int bestFeature = 0;
            double bestThreshold = 0;
            int bestLeftClass = y[0];
            int bestRightClass = y[0];
            
            // Find unique classes
            Set<Integer> uniqueClasses = new HashSet<>();
            for (int label : y) {
                uniqueClasses.add(label);
            }
            Integer[] classArray = uniqueClasses.toArray(new Integer[0]);
            
            // Try each feature
            for (int f = 0; f < nFeatures; f++) {
                // Get sorted unique values for this feature
                double[] values = new double[nSamples];
                for (int i = 0; i < nSamples; i++) {
                    values[i] = X[i][f];
                }
                Arrays.sort(values);
                
                // Try thresholds between consecutive values
                for (int i = 0; i < nSamples - 1; i++) {
                    if (values[i] == values[i + 1]) continue;
                    
                    double threshold = (values[i] + values[i + 1]) / 2.0;
                    
                    // Try both class assignments
                    for (int leftClass : classArray) {
                        for (int rightClass : classArray) {
                            double gain = calculateGain(X, y, weights, f, threshold, leftClass, rightClass);
                            
                            if (gain > bestGain) {
                                bestGain = gain;
                                bestFeature = f;
                                bestThreshold = threshold;
                                bestLeftClass = leftClass;
                                bestRightClass = rightClass;
                            }
                        }
                    }
                }
            }
            
            this.featureIndex = bestFeature;
            this.threshold = bestThreshold;
            this.leftClass = bestLeftClass;
            this.rightClass = bestRightClass;
        }
        
        private double calculateGain(double[][] X, int[] y, double[] weights,
                                      int feature, double threshold, int leftClass, int rightClass) {
            double correct = 0.0;
            
            for (int i = 0; i < X.length; i++) {
                int predicted = X[i][feature] <= threshold ? leftClass : rightClass;
                if (predicted == y[i]) {
                    correct += weights[i];
                }
            }
            
            return correct;
        }
        
        public int[] predict(double[][] X) {
            int[] predictions = new int[X.length];
            for (int i = 0; i < X.length; i++) {
                predictions[i] = X[i][featureIndex] <= threshold ? leftClass : rightClass;
            }
            return predictions;
        }
        
        public int getFeatureIndex() {
            return featureIndex;
        }
    }
}
