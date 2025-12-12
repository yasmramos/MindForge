package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * AdaBoost (Adaptive Boosting) Classifier.
 * 
 * An AdaBoost classifier is a meta-estimator that begins by fitting a classifier
 * on the original dataset and then fits additional copies of the classifier on the
 * same dataset but where the weights of incorrectly classified instances are adjusted
 * such that subsequent classifiers focus more on difficult cases.
 * 
 * This implementation supports:
 * - Binary and multiclass classification (using SAMME algorithm)
 * - Decision stumps (1-level decision trees) as weak learners
 * - Configurable learning rate for regularization
 * - Feature importance calculation
 * 
 * Example usage:
 * <pre>{@code
 * // Using Builder pattern
 * AdaBoostClassifier ada = new AdaBoostClassifier.Builder()
 *     .nEstimators(100)
 *     .learningRate(0.5)
 *     .randomState(42)
 *     .build();
 * 
 * ada.fit(X_train, y_train);
 * int[] predictions = ada.predict(X_test);
 * double[][] probas = ada.predictProba(X_test);
 * 
 * // Or using simple constructor
 * AdaBoostClassifier ada = new AdaBoostClassifier(50);
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.1
 */
public class AdaBoostClassifier implements Classifier<double[]>, Serializable {
    
    private static final long serialVersionUID = 2L;
    
    private final int nEstimators;
    private final double learningRate;
    private final Integer randomState;
    private final String algorithm; // "SAMME" or "SAMME.R"
    
    private List<DecisionStump> stumps;
    private List<Double> stumpWeights;
    private List<Double> stumpErrors;
    private int[] classes;
    private int nClasses;
    private int nFeatures;
    private boolean isFitted;
    private Random random;
    
    /**
     * Creates an AdaBoostClassifier with default settings.
     * Uses 50 estimators and learning rate of 1.0.
     */
    public AdaBoostClassifier() {
        this(50, 1.0, null, "SAMME");
    }
    
    /**
     * Creates an AdaBoostClassifier with specified number of estimators.
     * 
     * @param nEstimators The maximum number of estimators
     */
    public AdaBoostClassifier(int nEstimators) {
        this(nEstimators, 1.0, null, "SAMME");
    }
    
    /**
     * Creates an AdaBoostClassifier with full configuration (legacy constructor).
     * 
     * @param nEstimators The maximum number of estimators
     * @param learningRate Weight applied to each classifier at each boosting iteration
     * @param randomState Random seed for reproducibility (-1 for random)
     * @throws IllegalArgumentException if nEstimators < 1 or learningRate <= 0
     */
    public AdaBoostClassifier(int nEstimators, double learningRate, int randomState) {
        this(nEstimators, learningRate, randomState == -1 ? null : randomState, "SAMME");
    }
    
    /**
     * Private constructor used by Builder.
     */
    private AdaBoostClassifier(int nEstimators, double learningRate, 
                                Integer randomState, String algorithm) {
        if (nEstimators < 1) {
            throw new IllegalArgumentException("nEstimators must be at least 1, got: " + nEstimators);
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got: " + learningRate);
        }
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.randomState = randomState;
        this.algorithm = algorithm;
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
        this.random = randomState != null ? new Random(randomState) : new Random();
        
        // Find unique classes
        Set<Integer> uniqueClasses = new TreeSet<>();
        for (int label : y) {
            uniqueClasses.add(label);
        }
        this.classes = uniqueClasses.stream().mapToInt(Integer::intValue).toArray();
        this.nClasses = classes.length;
        
        int nSamples = X.length;
        double[] sampleWeights = new double[nSamples];
        Arrays.fill(sampleWeights, 1.0 / nSamples);
        
        this.stumps = new ArrayList<>();
        this.stumpWeights = new ArrayList<>();
        this.stumpErrors = new ArrayList<>();
        
        for (int t = 0; t < nEstimators; t++) {
            // Fit a decision stump
            DecisionStump stump = new DecisionStump();
            stump.fit(X, y, sampleWeights, classes, random);
            
            // Get predictions
            int[] predictions = stump.predict(X);
            
            // Calculate weighted error
            double error = 0.0;
            for (int i = 0; i < nSamples; i++) {
                if (predictions[i] != y[i]) {
                    error += sampleWeights[i];
                }
            }
            
            // Store error
            stumpErrors.add(error);
            
            // Avoid division by zero and log(0)
            error = Math.max(error, 1e-10);
            error = Math.min(error, 1.0 - 1e-10);
            
            // For multiclass SAMME algorithm
            double alpha;
            if (nClasses == 2) {
                // Binary classification
                if (error >= 0.5) {
                    if (stumps.isEmpty()) {
                        stumps.add(stump);
                        stumpWeights.add(1.0);
                    }
                    break;
                }
                alpha = learningRate * 0.5 * Math.log((1.0 - error) / error);
            } else {
                // Multiclass SAMME algorithm
                // Stop if error is worse than random guessing
                if (error >= 1.0 - 1.0 / nClasses) {
                    if (stumps.isEmpty()) {
                        stumps.add(stump);
                        stumpWeights.add(1.0);
                    }
                    break;
                }
                alpha = learningRate * (Math.log((1.0 - error) / error) + Math.log(nClasses - 1));
            }
            
            stumps.add(stump);
            stumpWeights.add(alpha);
            
            // Update sample weights
            double sumWeights = 0.0;
            for (int i = 0; i < nSamples; i++) {
                if (predictions[i] != y[i]) {
                    sampleWeights[i] *= Math.exp(alpha);
                }
                sumWeights += sampleWeights[i];
            }
            
            // Normalize weights
            for (int i = 0; i < nSamples; i++) {
                sampleWeights[i] /= sumWeights;
            }
            
            // Early stopping on perfect classification
            if (error < 1e-10) {
                break;
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
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted before predict");
        }
        if (x.length != nFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", nFeatures, x.length));
        }
        
        // Weighted vote for each class
        double[] classScores = new double[nClasses];
        
        for (int t = 0; t < stumps.size(); t++) {
            int pred = stumps.get(t).predict(x);
            int classIdx = findClassIndex(pred);
            if (classIdx >= 0) {
                classScores[classIdx] += stumpWeights.get(t);
            }
        }
        
        // Return class with highest score
        int maxIdx = 0;
        double maxScore = classScores[0];
        for (int i = 1; i < nClasses; i++) {
            if (classScores[i] > maxScore) {
                maxScore = classScores[i];
                maxIdx = i;
            }
        }
        
        return classes[maxIdx];
    }
    
    /**
     * Returns the number of classes this classifier can predict.
     * 
     * @return number of classes
     */
    @Override
    public int getNumClasses() {
        if (!isFitted) {
            return 0;
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
        
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Predict class probabilities for a single sample.
     * 
     * @param x Input sample
     * @return Class probabilities
     */
    public double[] predictProba(double[] x) {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted before predict");
        }
        
        // Weighted vote for each class
        double[] classScores = new double[nClasses];
        
        for (int t = 0; t < stumps.size(); t++) {
            int pred = stumps.get(t).predict(x);
            int classIdx = findClassIndex(pred);
            if (classIdx >= 0) {
                classScores[classIdx] += stumpWeights.get(t);
            }
        }
        
        // Convert to probabilities using softmax
        double maxScore = Double.NEGATIVE_INFINITY;
        for (double score : classScores) {
            maxScore = Math.max(maxScore, score);
        }
        
        double sumExp = 0.0;
        double[] proba = new double[nClasses];
        for (int i = 0; i < nClasses; i++) {
            proba[i] = Math.exp(classScores[i] - maxScore);
            sumExp += proba[i];
        }
        
        for (int i = 0; i < nClasses; i++) {
            proba[i] /= sumExp;
        }
        
        return proba;
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
        
        double[][] proba = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            proba[i] = predictProba(X[i]);
        }
        return proba;
    }
    
    /**
     * Returns the decision function (weighted scores) for a sample.
     * 
     * @param x Input sample
     * @return Weighted scores for each class
     */
    public double[] decisionFunction(double[] x) {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted before predict");
        }
        
        double[] classScores = new double[nClasses];
        
        for (int t = 0; t < stumps.size(); t++) {
            int pred = stumps.get(t).predict(x);
            int classIdx = findClassIndex(pred);
            if (classIdx >= 0) {
                classScores[classIdx] += stumpWeights.get(t);
            }
        }
        
        return classScores;
    }
    
    /**
     * Computes accuracy score on given data.
     * 
     * @param X Input features
     * @param y True labels
     * @return Accuracy score
     */
    public double score(double[][] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        return (double) correct / y.length;
    }
    
    private int findClassIndex(int classLabel) {
        for (int i = 0; i < classes.length; i++) {
            if (classes[i] == classLabel) {
                return i;
            }
        }
        return -1;
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
     * Alias for getActualNEstimators for consistency.
     * 
     * @return number of estimators used
     */
    public int getNumEstimators() {
        return getActualNEstimators();
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
     * Get the error of each estimator.
     * 
     * @return array of estimator errors
     */
    public double[] getEstimatorErrors() {
        if (!isFitted) {
            throw new IllegalStateException("AdaBoostClassifier must be fitted first");
        }
        return stumpErrors.stream().mapToDouble(Double::doubleValue).toArray();
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
     * Alias for isFitted for interface consistency.
     * 
     * @return true if trained
     */
    public boolean isTrained() {
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
     * Get the algorithm used.
     * 
     * @return algorithm name ("SAMME" or "SAMME.R")
     */
    public String getAlgorithm() {
        return algorithm;
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
        
        public void fit(double[][] X, int[] y, double[] weights, int[] classes, Random random) {
            int nSamples = X.length;
            int nFeatures = X[0].length;
            
            double bestGain = Double.NEGATIVE_INFINITY;
            int bestFeature = 0;
            double bestThreshold = 0;
            int bestLeftClass = classes[0];
            int bestRightClass = classes[0];
            
            // Randomly select subset of features for diversity
            int nTryFeatures = Math.max(1, (int) Math.sqrt(nFeatures));
            List<Integer> featureIndices = new ArrayList<>();
            for (int i = 0; i < nFeatures; i++) {
                featureIndices.add(i);
            }
            Collections.shuffle(featureIndices, random);
            
            // Try each selected feature
            for (int fi = 0; fi < Math.min(nTryFeatures, nFeatures); fi++) {
                int f = featureIndices.get(fi);
                
                // Get sorted unique values for this feature
                Set<Double> uniqueValues = new TreeSet<>();
                for (int i = 0; i < nSamples; i++) {
                    uniqueValues.add(X[i][f]);
                }
                Double[] sortedValues = uniqueValues.toArray(new Double[0]);
                
                // Try thresholds between consecutive values
                for (int i = 0; i < sortedValues.length - 1; i++) {
                    double threshold = (sortedValues[i] + sortedValues[i + 1]) / 2.0;
                    
                    // Try all class assignments
                    for (int leftClass : classes) {
                        for (int rightClass : classes) {
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
        
        public int predict(double[] x) {
            return x[featureIndex] <= threshold ? leftClass : rightClass;
        }
        
        public int[] predict(double[][] X) {
            int[] predictions = new int[X.length];
            for (int i = 0; i < X.length; i++) {
                predictions[i] = predict(X[i]);
            }
            return predictions;
        }
        
        public int getFeatureIndex() {
            return featureIndex;
        }
    }
    
    /**
     * Builder class for AdaBoostClassifier.
     */
    public static class Builder {
        private int nEstimators = 50;
        private double learningRate = 1.0;
        private Integer randomState = null;
        private String algorithm = "SAMME";
        
        /**
         * Sets the number of boosting iterations (weak learners).
         * 
         * @param nEstimators Number of estimators (default: 50)
         * @return this builder
         */
        public Builder nEstimators(int nEstimators) {
            if (nEstimators < 1) {
                throw new IllegalArgumentException("nEstimators must be at least 1");
            }
            this.nEstimators = nEstimators;
            return this;
        }
        
        /**
         * Sets the learning rate (shrinkage).
         * Lower values require more estimators but can achieve better generalization.
         * 
         * @param learningRate Learning rate (default: 1.0)
         * @return this builder
         */
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0.0) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        /**
         * Sets the boosting algorithm.
         * 
         * @param algorithm Either "SAMME" (default) or "SAMME.R"
         * @return this builder
         */
        public Builder algorithm(String algorithm) {
            if (!algorithm.equals("SAMME") && !algorithm.equals("SAMME.R")) {
                throw new IllegalArgumentException("algorithm must be 'SAMME' or 'SAMME.R'");
            }
            this.algorithm = algorithm;
            return this;
        }
        
        /**
         * Sets the random state for reproducibility.
         * 
         * @param randomState Random seed
         * @return this builder
         */
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        /**
         * Builds the AdaBoostClassifier instance.
         * 
         * @return new AdaBoostClassifier
         */
        public AdaBoostClassifier build() {
            return new AdaBoostClassifier(nEstimators, learningRate, randomState, algorithm);
        }
    }
}
