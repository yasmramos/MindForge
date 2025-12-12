package io.github.yasmramos.mindforge.classification;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * Random Forest Classifier - An ensemble learning method using multiple decision trees.
 * 
 * Random Forest builds multiple decision trees using bootstrap samples of the training data
 * and random feature selection at each split. The final prediction is made by majority voting
 * across all trees. This approach reduces overfitting and improves generalization.
 * 
 * Key Features:
 * - Bootstrap aggregating (bagging) for creating diverse trees
 * - Random feature selection at each split for decorrelation
 * - Out-of-bag (OOB) score for validation without separate test set
 * - Feature importance calculation
 * - Parallel tree building for efficiency
 * - Configurable ensemble size and tree parameters
 * 
 * Example usage:
 * <pre>
 * RandomForestClassifier rf = new RandomForestClassifier.Builder()
 *     .nEstimators(100)
 *     .maxFeatures("sqrt")
 *     .maxDepth(15)
 *     .minSamplesSplit(2)
 *     .build();
 * 
 * rf.fit(X_train, y_train);
 * int[] predictions = rf.predict(X_test);
 * double[][] probabilities = rf.predictProba(X_test);
 * double oobScore = rf.getOOBScore();
 * double[] importance = rf.getFeatureImportance();
 * </pre>
 * 
 * @author MindForge
 * @version 1.0.3-alpha
 */
public class RandomForestClassifier {
    
    private List<DecisionTreeClassifier> trees;
    private int nEstimators;
    private int maxFeatures;
    private String maxFeaturesMode; // "sqrt", "log2", or integer
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private DecisionTreeClassifier.Criterion criterion;
    private boolean bootstrap;
    private Random random;
    private int randomState;
    
    // Training data stored for OOB score calculation
    private double[][] X_train;
    private int[] y_train;
    private int nFeatures;
    private int[] classes;
    
    // Out-of-bag predictions
    private Map<Integer, List<Integer>> oobPredictions;
    private double oobScore;
    
    // Feature importance
    private double[] featureImportance;
    
    /**
     * Builder pattern for constructing RandomForestClassifier with custom parameters.
     */
    public static class Builder {
        private int nEstimators = 100;
        private String maxFeaturesMode = "sqrt";
        private int maxDepth = Integer.MAX_VALUE;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private DecisionTreeClassifier.Criterion criterion = DecisionTreeClassifier.Criterion.GINI;
        private boolean bootstrap = true;
        private int randomState = 42;
        
        /**
         * Set the number of trees in the forest.
         * @param nEstimators Number of decision trees (default: 100)
         * @return Builder instance
         */
        public Builder nEstimators(int nEstimators) {
            if (nEstimators <= 0) {
                throw new IllegalArgumentException("nEstimators must be positive");
            }
            this.nEstimators = nEstimators;
            return this;
        }
        
        /**
         * Set the number of features to consider when looking for the best split.
         * @param maxFeatures "sqrt", "log2", or a specific number
         * @return Builder instance
         */
        public Builder maxFeatures(String maxFeatures) {
            if (!maxFeatures.equals("sqrt") && !maxFeatures.equals("log2")) {
                throw new IllegalArgumentException("maxFeatures must be 'sqrt' or 'log2'");
            }
            this.maxFeaturesMode = maxFeatures;
            return this;
        }
        
        /**
         * Set the number of features to consider when looking for the best split.
         * @param maxFeatures Number of features
         * @return Builder instance
         */
        public Builder maxFeatures(int maxFeatures) {
            if (maxFeatures <= 0) {
                throw new IllegalArgumentException("maxFeatures must be positive");
            }
            this.maxFeaturesMode = String.valueOf(maxFeatures);
            return this;
        }
        
        /**
         * Set the maximum depth of each tree.
         * @param maxDepth Maximum tree depth (default: unlimited)
         * @return Builder instance
         */
        public Builder maxDepth(int maxDepth) {
            if (maxDepth <= 0) {
                throw new IllegalArgumentException("maxDepth must be positive");
            }
            this.maxDepth = maxDepth;
            return this;
        }
        
        /**
         * Set the minimum number of samples required to split an internal node.
         * @param minSamplesSplit Minimum samples for split (default: 2)
         * @return Builder instance
         */
        public Builder minSamplesSplit(int minSamplesSplit) {
            if (minSamplesSplit < 2) {
                throw new IllegalArgumentException("minSamplesSplit must be at least 2");
            }
            this.minSamplesSplit = minSamplesSplit;
            return this;
        }
        
        /**
         * Set the minimum number of samples required to be at a leaf node.
         * @param minSamplesLeaf Minimum samples at leaf (default: 1)
         * @return Builder instance
         */
        public Builder minSamplesLeaf(int minSamplesLeaf) {
            if (minSamplesLeaf < 1) {
                throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
            }
            this.minSamplesLeaf = minSamplesLeaf;
            return this;
        }
        
        /**
         * Set the function to measure the quality of a split.
         * @param criterion Splitting criterion (GINI or ENTROPY)
         * @return Builder instance
         */
        public Builder criterion(DecisionTreeClassifier.Criterion criterion) {
            this.criterion = criterion;
            return this;
        }
        
        /**
         * Set whether to use bootstrap samples when building trees.
         * @param bootstrap Use bootstrap sampling (default: true)
         * @return Builder instance
         */
        public Builder bootstrap(boolean bootstrap) {
            this.bootstrap = bootstrap;
            return this;
        }
        
        /**
         * Set the random seed for reproducibility.
         * @param randomState Random seed
         * @return Builder instance
         */
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        /**
         * Build the RandomForestClassifier with the configured parameters.
         * @return Configured RandomForestClassifier instance
         */
        public RandomForestClassifier build() {
            return new RandomForestClassifier(this);
        }
    }
    
    /**
     * Private constructor used by Builder.
     */
    private RandomForestClassifier(Builder builder) {
        this.nEstimators = builder.nEstimators;
        this.maxFeaturesMode = builder.maxFeaturesMode;
        this.maxDepth = builder.maxDepth;
        this.minSamplesSplit = builder.minSamplesSplit;
        this.minSamplesLeaf = builder.minSamplesLeaf;
        this.criterion = builder.criterion;
        this.bootstrap = builder.bootstrap;
        this.randomState = builder.randomState;
        this.random = new Random(randomState);
        this.trees = new ArrayList<>(nEstimators);
        this.oobPredictions = new HashMap<>();
    }
    
    /**
     * Calculate the number of features to use based on maxFeaturesMode.
     */
    private int calculateMaxFeatures(int totalFeatures) {
        if (maxFeaturesMode.equals("sqrt")) {
            return (int) Math.sqrt(totalFeatures);
        } else if (maxFeaturesMode.equals("log2")) {
            return (int) (Math.log(totalFeatures) / Math.log(2));
        } else {
            int value = Integer.parseInt(maxFeaturesMode);
            return Math.min(value, totalFeatures);
        }
    }
    
    /**
     * Create a bootstrap sample of the training data.
     */
    private int[] createBootstrapSample(int nSamples) {
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = random.nextInt(nSamples);
        }
        return indices;
    }
    
    /**
     * Get out-of-bag indices (samples not in bootstrap sample).
     */
    private Set<Integer> getOOBIndices(int[] bootstrapIndices, int nSamples) {
        Set<Integer> inBag = new HashSet<>();
        for (int idx : bootstrapIndices) {
            inBag.add(idx);
        }
        
        Set<Integer> oob = new HashSet<>();
        for (int i = 0; i < nSamples; i++) {
            if (!inBag.contains(i)) {
                oob.add(i);
            }
        }
        return oob;
    }
    
    /**
     * Extract samples based on indices.
     */
    private double[][] extractSamples(double[][] X, int[] indices) {
        double[][] samples = new double[indices.length][X[0].length];
        for (int i = 0; i < indices.length; i++) {
            samples[i] = X[indices[i]].clone();
        }
        return samples;
    }
    
    /**
     * Extract labels based on indices.
     */
    private int[] extractLabels(int[] y, int[] indices) {
        int[] labels = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            labels[i] = y[indices[i]];
        }
        return labels;
    }
    
    /**
     * Train the Random Forest classifier.
     * 
     * @param X Training feature matrix (n_samples × n_features)
     * @param y Training labels (n_samples)
     */
    public void fit(double[][] X, int[] y) {
        if (X == null || y == null || X.length == 0 || y.length == 0) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
        
        if (X.length != y.length) {
            throw new IllegalArgumentException("Number of samples in X and y must match");
        }
        
        this.X_train = X;
        this.y_train = y;
        this.nFeatures = X[0].length;
        this.maxFeatures = calculateMaxFeatures(nFeatures);
        
        // Get unique classes
        Set<Integer> classSet = new HashSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        this.classes = classSet.stream().mapToInt(Integer::intValue).sorted().toArray();
        
        // Initialize OOB predictions storage
        oobPredictions.clear();
        
        // Build trees in parallel
        int nSamples = X.length;
        trees = new ArrayList<>(nEstimators);
        
        // Use parallel stream for efficiency
        List<TreeData> treeDataList = IntStream.range(0, nEstimators)
            .parallel()
            .mapToObj(i -> buildSingleTree(X, y, nSamples, i))
            .collect(java.util.stream.Collectors.toList());
        
        // Extract trees and update OOB predictions
        for (TreeData treeData : treeDataList) {
            trees.add(treeData.tree);
            updateOOBPredictions(treeData.tree, treeData.oobIndices, treeData.treeIndex);
        }
        
        // Calculate OOB score
        calculateOOBScore();
        
        // Calculate feature importance
        calculateFeatureImportance();
    }
    
    /**
     * Helper class to store tree data during parallel processing.
     */
    private static class TreeData {
        DecisionTreeClassifier tree;
        Set<Integer> oobIndices;
        int treeIndex;
        
        TreeData(DecisionTreeClassifier tree, Set<Integer> oobIndices, int treeIndex) {
            this.tree = tree;
            this.oobIndices = oobIndices;
            this.treeIndex = treeIndex;
        }
    }
    
    /**
     * Build a single decision tree.
     */
    private TreeData buildSingleTree(double[][] X, int[] y, int nSamples, int treeIndex) {
        Random treeRandom = new Random(randomState + treeIndex);
        
        int[] bootstrapIndices;
        Set<Integer> oobIndices;
        
        if (bootstrap) {
            // Create bootstrap sample
            bootstrapIndices = createBootstrapSample(nSamples);
            oobIndices = getOOBIndices(bootstrapIndices, nSamples);
        } else {
            // Use all samples
            bootstrapIndices = IntStream.range(0, nSamples).toArray();
            oobIndices = new HashSet<>();
        }
        
        // Extract bootstrap samples
        double[][] X_bootstrap = extractSamples(X, bootstrapIndices);
        int[] y_bootstrap = extractLabels(y, bootstrapIndices);
        
        // Build decision tree with random feature selection
        DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
            .maxDepth(maxDepth)
            .minSamplesSplit(minSamplesSplit)
            .minSamplesLeaf(minSamplesLeaf)
            .criterion(criterion)
            .maxFeatures(maxFeatures)
            .randomState(randomState + treeIndex)
            .build();
        
        tree.fit(X_bootstrap, y_bootstrap);
        
        return new TreeData(tree, oobIndices, treeIndex);
    }
    
    /**
     * Update OOB predictions for samples not in this tree's bootstrap sample.
     */
    private synchronized void updateOOBPredictions(DecisionTreeClassifier tree, Set<Integer> oobIndices, int treeIndex) {
        for (int idx : oobIndices) {
            double[][] sample = new double[][] { X_train[idx] };
            int prediction = tree.predict(sample)[0];
            
            oobPredictions.computeIfAbsent(idx, k -> new ArrayList<>()).add(prediction);
        }
    }
    
    /**
     * Calculate out-of-bag score.
     */
    private void calculateOOBScore() {
        if (!bootstrap || oobPredictions.isEmpty()) {
            oobScore = Double.NaN;
            return;
        }
        
        int correct = 0;
        int total = 0;
        
        for (Map.Entry<Integer, List<Integer>> entry : oobPredictions.entrySet()) {
            int idx = entry.getKey();
            List<Integer> predictions = entry.getValue();
            
            if (predictions.isEmpty()) {
                continue;
            }
            
            // Majority vote
            int prediction = getMajorityVote(predictions);
            if (prediction == y_train[idx]) {
                correct++;
            }
            total++;
        }
        
        oobScore = total > 0 ? (double) correct / total : Double.NaN;
    }
    
    /**
     * Get majority vote from predictions.
     */
    private int getMajorityVote(List<Integer> predictions) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int pred : predictions) {
            counts.put(pred, counts.getOrDefault(pred, 0) + 1);
        }
        
        return counts.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(predictions.get(0));
    }
    
    /**
     * Calculate feature importance as average across all trees.
     */
    private void calculateFeatureImportance() {
        featureImportance = new double[nFeatures];
        
        for (DecisionTreeClassifier tree : trees) {
            double[] treeImportance = tree.getFeatureImportance();
            if (treeImportance != null) {
                for (int i = 0; i < nFeatures; i++) {
                    featureImportance[i] += treeImportance[i];
                }
            }
        }
        
        // Normalize
        for (int i = 0; i < nFeatures; i++) {
            featureImportance[i] /= nEstimators;
        }
    }
    
    /**
     * Make predictions for test samples.
     * 
     * @param X Test feature matrix (n_samples × n_features)
     * @return Predicted class labels
     */
    public int[] predict(double[][] X) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Test data cannot be null or empty");
        }
        
        int[] predictions = new int[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double[][] sample = new double[][] { X[i] };
            List<Integer> treePredictions = new ArrayList<>(nEstimators);
            
            for (DecisionTreeClassifier tree : trees) {
                treePredictions.add(tree.predict(sample)[0]);
            }
            
            predictions[i] = getMajorityVote(treePredictions);
        }
        
        return predictions;
    }
    
    /**
     * Predict class probabilities for test samples.
     * 
     * @param X Test feature matrix (n_samples × n_features)
     * @return Predicted class probabilities (n_samples × n_classes)
     */
    public double[][] predictProba(double[][] X) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("Test data cannot be null or empty");
        }
        
        double[][] probabilities = new double[X.length][classes.length];
        
        for (int i = 0; i < X.length; i++) {
            double[][] sample = new double[][] { X[i] };
            
            // Collect probabilities from all trees
            for (DecisionTreeClassifier tree : trees) {
                double[][] treeProba = tree.predictProba(sample);
                // Handle trees that may have seen fewer classes in their bootstrap sample
                int treeProbLength = treeProba[0].length;
                for (int j = 0; j < Math.min(classes.length, treeProbLength); j++) {
                    probabilities[i][j] += treeProba[0][j];
                }
            }
            
            // Average probabilities and normalize
            double sum = 0.0;
            for (int j = 0; j < classes.length; j++) {
                probabilities[i][j] /= nEstimators;
                sum += probabilities[i][j];
            }
            
            // Normalize to ensure probabilities sum to 1
            if (sum > 0) {
                for (int j = 0; j < classes.length; j++) {
                    probabilities[i][j] /= sum;
                }
            }
        }
        
        return probabilities;
    }
    
    /**
     * Get the out-of-bag score.
     * 
     * @return OOB score (accuracy on out-of-bag samples)
     */
    public double getOOBScore() {
        return oobScore;
    }
    
    /**
     * Get feature importance scores.
     * 
     * @return Array of feature importance values
     */
    public double[] getFeatureImportance() {
        return featureImportance != null ? featureImportance.clone() : null;
    }
    
    /**
     * Get the number of trees in the forest.
     * 
     * @return Number of estimators
     */
    public int getNEstimators() {
        return nEstimators;
    }
    
    /**
     * Get the list of unique classes.
     * 
     * @return Array of class labels
     */
    public int[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
}
