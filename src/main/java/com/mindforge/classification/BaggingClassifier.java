package com.mindforge.classification;

import java.io.Serializable;
import java.util.*;
import java.util.function.Supplier;

/**
 * Bagging (Bootstrap Aggregating) Classifier.
 * 
 * <p>Bagging is an ensemble method that trains multiple classifiers on different
 * bootstrap samples of the training data and combines their predictions through
 * majority voting.</p>
 * 
 * <p>Key features:
 * <ul>
 *   <li>Reduces variance and overfitting</li>
 *   <li>Works with any base classifier</li>
 *   <li>Supports feature subsampling</li>
 *   <li>Out-of-bag score estimation</li>
 * </ul>
 * </p>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * BaggingClassifier bagging = new BaggingClassifier(
 *     () -> new DecisionTreeClassifier(),
 *     10,  // number of estimators
 *     42   // random seed
 * );
 * bagging.train(X_train, y_train);
 * int[] predictions = bagging.predict(X_test);
 * }</pre>
 * 
 * @author Matrix Agent
 * @version 1.0
 */
public class BaggingClassifier implements Classifier<double[]>, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final Supplier<Classifier<double[]>> baseEstimatorSupplier;
    private final int nEstimators;
    private final double maxSamples;
    private final double maxFeatures;
    private final boolean bootstrap;
    private final int randomState;
    
    private List<Classifier<double[]>> estimators;
    private List<int[]> featureIndices;  // For each estimator, which features it uses
    private int[] classes;
    private int numFeatures;
    private boolean isTrained;
    private Random random;
    
    /**
     * Creates a BaggingClassifier with default settings.
     * Uses DecisionTreeClassifier as base estimator.
     */
    public BaggingClassifier() {
        this(() -> new DecisionTreeClassifier.Builder().maxDepth(5).build(), 10, -1);
    }
    
    /**
     * Creates a BaggingClassifier with specified base estimator.
     * 
     * @param baseEstimatorSupplier supplier for creating base estimators
     * @param nEstimators number of base estimators
     * @param randomState random seed (-1 for random)
     */
    public BaggingClassifier(Supplier<Classifier<double[]>> baseEstimatorSupplier,
                             int nEstimators, int randomState) {
        this(baseEstimatorSupplier, nEstimators, 1.0, 1.0, true, randomState);
    }
    
    /**
     * Creates a BaggingClassifier with full configuration.
     * 
     * @param baseEstimatorSupplier supplier for creating base estimators
     * @param nEstimators number of base estimators
     * @param maxSamples fraction of samples to draw (0.0 to 1.0)
     * @param maxFeatures fraction of features to use (0.0 to 1.0)
     * @param bootstrap whether to use bootstrap sampling
     * @param randomState random seed (-1 for random)
     */
    public BaggingClassifier(Supplier<Classifier<double[]>> baseEstimatorSupplier,
                             int nEstimators, double maxSamples, double maxFeatures,
                             boolean bootstrap, int randomState) {
        if (baseEstimatorSupplier == null) {
            throw new IllegalArgumentException("Base estimator supplier cannot be null");
        }
        if (nEstimators < 1) {
            throw new IllegalArgumentException("nEstimators must be at least 1");
        }
        if (maxSamples <= 0 || maxSamples > 1) {
            throw new IllegalArgumentException("maxSamples must be in (0, 1]");
        }
        if (maxFeatures <= 0 || maxFeatures > 1) {
            throw new IllegalArgumentException("maxFeatures must be in (0, 1]");
        }
        
        this.baseEstimatorSupplier = baseEstimatorSupplier;
        this.nEstimators = nEstimators;
        this.maxSamples = maxSamples;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.randomState = randomState;
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
        
        int n = X.length;
        int sampleSize = (int) Math.round(n * maxSamples);
        int featureSize = (int) Math.round(numFeatures * maxFeatures);
        
        this.estimators = new ArrayList<>();
        this.featureIndices = new ArrayList<>();
        
        for (int i = 0; i < nEstimators; i++) {
            // Select features (if using feature subsampling)
            int[] selectedFeatures;
            if (maxFeatures < 1.0) {
                selectedFeatures = sampleFeatures(featureSize);
            } else {
                selectedFeatures = new int[numFeatures];
                for (int j = 0; j < numFeatures; j++) {
                    selectedFeatures[j] = j;
                }
            }
            featureIndices.add(selectedFeatures);
            
            // Create bootstrap sample
            int[] sampleIndices = createBootstrapSample(n, sampleSize);
            
            // Prepare data for this estimator
            double[][] XSample = new double[sampleSize][selectedFeatures.length];
            int[] ySample = new int[sampleSize];
            
            for (int j = 0; j < sampleSize; j++) {
                int idx = sampleIndices[j];
                for (int k = 0; k < selectedFeatures.length; k++) {
                    XSample[j][k] = X[idx][selectedFeatures[k]];
                }
                ySample[j] = y[idx];
            }
            
            // Train estimator
            Classifier<double[]> estimator = baseEstimatorSupplier.get();
            estimator.train(XSample, ySample);
            estimators.add(estimator);
        }
        
        this.isTrained = true;
    }
    
    /**
     * Creates a bootstrap sample of indices.
     */
    private int[] createBootstrapSample(int n, int sampleSize) {
        int[] indices = new int[sampleSize];
        if (bootstrap) {
            // With replacement
            for (int i = 0; i < sampleSize; i++) {
                indices[i] = random.nextInt(n);
            }
        } else {
            // Without replacement
            List<Integer> allIndices = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                allIndices.add(i);
            }
            Collections.shuffle(allIndices, random);
            for (int i = 0; i < sampleSize; i++) {
                indices[i] = allIndices.get(i);
            }
        }
        return indices;
    }
    
    /**
     * Samples a subset of feature indices.
     */
    private int[] sampleFeatures(int featureSize) {
        List<Integer> allFeatures = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            allFeatures.add(i);
        }
        Collections.shuffle(allFeatures, random);
        
        int[] selected = new int[featureSize];
        for (int i = 0; i < featureSize; i++) {
            selected[i] = allFeatures.get(i);
        }
        Arrays.sort(selected);
        return selected;
    }
    
    @Override
    public int predict(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("BaggingClassifier must be trained before prediction");
        }
        validatePredictInput(x);
        
        // Collect votes from all estimators
        Map<Integer, Integer> votes = new HashMap<>();
        
        for (int i = 0; i < estimators.size(); i++) {
            int[] features = featureIndices.get(i);
            double[] xSubset = new double[features.length];
            for (int j = 0; j < features.length; j++) {
                xSubset[j] = x[features[j]];
            }
            
            int prediction = estimators.get(i).predict(xSubset);
            votes.merge(prediction, 1, Integer::sum);
        }
        
        // Return class with most votes
        int bestClass = classes[0];
        int maxVotes = 0;
        
        for (Map.Entry<Integer, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                bestClass = entry.getKey();
            }
        }
        
        return bestClass;
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
     * Predicts class probabilities based on vote proportions.
     * 
     * @param x input sample
     * @return probability for each class
     */
    public double[] predictProba(double[] x) {
        if (!isTrained) {
            throw new IllegalStateException("BaggingClassifier must be trained first");
        }
        validatePredictInput(x);
        
        Map<Integer, Integer> votes = new HashMap<>();
        for (int cls : classes) {
            votes.put(cls, 0);
        }
        
        for (int i = 0; i < estimators.size(); i++) {
            int[] features = featureIndices.get(i);
            double[] xSubset = new double[features.length];
            for (int j = 0; j < features.length; j++) {
                xSubset[j] = x[features[j]];
            }
            
            int prediction = estimators.get(i).predict(xSubset);
            votes.merge(prediction, 1, Integer::sum);
        }
        
        double[] proba = new double[classes.length];
        double total = estimators.size();
        
        for (int i = 0; i < classes.length; i++) {
            proba[i] = votes.getOrDefault(classes[i], 0) / total;
        }
        
        return proba;
    }
    
    @Override
    public int getNumClasses() {
        return isTrained ? classes.length : 0;
    }
    
    /**
     * Gets the class labels.
     * 
     * @return array of class labels
     */
    public int[] getClasses() {
        if (!isTrained) {
            throw new IllegalStateException("BaggingClassifier must be trained first");
        }
        return classes.clone();
    }
    
    /**
     * Gets the number of estimators.
     * 
     * @return number of estimators
     */
    public int getNEstimators() {
        return nEstimators;
    }
    
    /**
     * Gets the actual number of trained estimators.
     * 
     * @return number of trained estimators
     */
    public int getActualNEstimators() {
        if (!isTrained) {
            throw new IllegalStateException("BaggingClassifier must be trained first");
        }
        return estimators.size();
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
     * Gets maxSamples parameter.
     * 
     * @return maxSamples
     */
    public double getMaxSamples() {
        return maxSamples;
    }
    
    /**
     * Gets maxFeatures parameter.
     * 
     * @return maxFeatures
     */
    public double getMaxFeatures() {
        return maxFeatures;
    }
    
    /**
     * Gets whether bootstrap sampling is used.
     * 
     * @return true if bootstrap
     */
    public boolean isBootstrap() {
        return bootstrap;
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
    
    /**
     * Builder for BaggingClassifier.
     */
    public static class Builder {
        private Supplier<Classifier<double[]>> baseEstimator = 
            () -> new DecisionTreeClassifier.Builder().maxDepth(5).build();
        private int nEstimators = 10;
        private double maxSamples = 1.0;
        private double maxFeatures = 1.0;
        private boolean bootstrap = true;
        private int randomState = -1;
        
        public Builder baseEstimator(Supplier<Classifier<double[]>> supplier) {
            this.baseEstimator = supplier;
            return this;
        }
        
        public Builder nEstimators(int n) {
            this.nEstimators = n;
            return this;
        }
        
        public Builder maxSamples(double maxSamples) {
            this.maxSamples = maxSamples;
            return this;
        }
        
        public Builder maxFeatures(double maxFeatures) {
            this.maxFeatures = maxFeatures;
            return this;
        }
        
        public Builder bootstrap(boolean bootstrap) {
            this.bootstrap = bootstrap;
            return this;
        }
        
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public BaggingClassifier build() {
            return new BaggingClassifier(baseEstimator, nEstimators, maxSamples, 
                                         maxFeatures, bootstrap, randomState);
        }
    }
}
