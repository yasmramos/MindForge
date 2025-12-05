package com.mindforge.regression;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

/**
 * Random Forest Regressor - an ensemble of Decision Tree Regressors.
 * 
 * <p>Random Forest builds multiple decision trees using bootstrap samples and
 * random feature subsets, then averages their predictions. This approach reduces
 * variance and prevents overfitting compared to single decision trees.</p>
 * 
 * <p>Features:</p>
 * <ul>
 *   <li>Bootstrap sampling (bagging)</li>
 *   <li>Random feature selection at each split</li>
 *   <li>Out-of-bag (OOB) error estimation</li>
 *   <li>Feature importance calculation</li>
 *   <li>Parallel training and prediction</li>
 *   <li>Configurable number of trees and tree parameters</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * double[][] X = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
 * double[] y = {1.5, 2.5, 3.5, 4.5};
 * 
 * RandomForestRegressor rf = new RandomForestRegressor.Builder()
 *     .nEstimators(100)
 *     .maxDepth(10)
 *     .maxFeatures("sqrt")
 *     .build();
 * 
 * rf.fit(X, y);
 * double prediction = rf.predict(new double[]{2.5, 3.5});
 * double oobScore = rf.getOobScore();
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class RandomForestRegressor implements Regressor<double[]> {
    
    /**
     * Strategy for selecting max features at each split.
     */
    public enum MaxFeaturesStrategy {
        /** Use all features */
        ALL,
        /** Use sqrt(n_features) */
        SQRT,
        /** Use log2(n_features) */
        LOG2,
        /** Use a specific fraction of features */
        FRACTION
    }
    
    // Hyperparameters
    private final int nEstimators;
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final MaxFeaturesStrategy maxFeaturesStrategy;
    private final double maxFeaturesFraction;
    private final boolean bootstrap;
    private final boolean oobScore;
    private final int nJobs;
    private final int randomState;
    private final boolean warmStart;
    private final double minImpurityDecrease;
    private final DecisionTreeRegressor.Criterion criterion;
    
    // Model state
    private List<DecisionTreeRegressor> trees;
    private double[] featureImportance;
    private double oobScoreValue;
    private double[][] oobPredictions;
    private int[] oobCounts;
    private int numFeatures;
    private boolean fitted;
    private Random random;
    private ExecutorService executor;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private RandomForestRegressor(Builder builder) {
        this.nEstimators = builder.nEstimators;
        this.maxDepth = builder.maxDepth;
        this.minSamplesSplit = builder.minSamplesSplit;
        this.minSamplesLeaf = builder.minSamplesLeaf;
        this.maxFeaturesStrategy = builder.maxFeaturesStrategy;
        this.maxFeaturesFraction = builder.maxFeaturesFraction;
        this.bootstrap = builder.bootstrap;
        this.oobScore = builder.oobScore;
        this.nJobs = builder.nJobs;
        this.randomState = builder.randomState;
        this.warmStart = builder.warmStart;
        this.minImpurityDecrease = builder.minImpurityDecrease;
        this.criterion = builder.criterion;
        this.fitted = false;
        this.random = new Random(randomState);
        this.trees = new ArrayList<>();
    }
    
    /**
     * Builder pattern for creating RandomForestRegressor instances.
     */
    public static class Builder {
        private int nEstimators = 100;
        private int maxDepth = Integer.MAX_VALUE;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private MaxFeaturesStrategy maxFeaturesStrategy = MaxFeaturesStrategy.SQRT;
        private double maxFeaturesFraction = 1.0;
        private boolean bootstrap = true;
        private boolean oobScore = false;
        private int nJobs = 1;
        private int randomState = 42;
        private boolean warmStart = false;
        private double minImpurityDecrease = 0.0;
        private DecisionTreeRegressor.Criterion criterion = DecisionTreeRegressor.Criterion.MSE;
        
        public Builder nEstimators(int nEstimators) {
            if (nEstimators < 1) {
                throw new IllegalArgumentException("nEstimators must be at least 1");
            }
            this.nEstimators = nEstimators;
            return this;
        }
        
        public Builder maxDepth(int maxDepth) {
            if (maxDepth < 1) {
                throw new IllegalArgumentException("maxDepth must be at least 1");
            }
            this.maxDepth = maxDepth;
            return this;
        }
        
        public Builder minSamplesSplit(int minSamplesSplit) {
            if (minSamplesSplit < 2) {
                throw new IllegalArgumentException("minSamplesSplit must be at least 2");
            }
            this.minSamplesSplit = minSamplesSplit;
            return this;
        }
        
        public Builder minSamplesLeaf(int minSamplesLeaf) {
            if (minSamplesLeaf < 1) {
                throw new IllegalArgumentException("minSamplesLeaf must be at least 1");
            }
            this.minSamplesLeaf = minSamplesLeaf;
            return this;
        }
        
        public Builder maxFeatures(String strategy) {
            switch (strategy.toLowerCase()) {
                case "auto":
                case "sqrt":
                    this.maxFeaturesStrategy = MaxFeaturesStrategy.SQRT;
                    break;
                case "log2":
                    this.maxFeaturesStrategy = MaxFeaturesStrategy.LOG2;
                    break;
                case "all":
                case "none":
                    this.maxFeaturesStrategy = MaxFeaturesStrategy.ALL;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown maxFeatures strategy: " + strategy);
            }
            return this;
        }
        
        public Builder maxFeatures(double fraction) {
            if (fraction <= 0 || fraction > 1) {
                throw new IllegalArgumentException("maxFeatures fraction must be in (0, 1]");
            }
            this.maxFeaturesStrategy = MaxFeaturesStrategy.FRACTION;
            this.maxFeaturesFraction = fraction;
            return this;
        }
        
        public Builder bootstrap(boolean bootstrap) {
            this.bootstrap = bootstrap;
            return this;
        }
        
        public Builder oobScore(boolean oobScore) {
            this.oobScore = oobScore;
            return this;
        }
        
        public Builder nJobs(int nJobs) {
            this.nJobs = nJobs;
            return this;
        }
        
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public Builder warmStart(boolean warmStart) {
            this.warmStart = warmStart;
            return this;
        }
        
        public Builder minImpurityDecrease(double minImpurityDecrease) {
            this.minImpurityDecrease = minImpurityDecrease;
            return this;
        }
        
        public Builder criterion(DecisionTreeRegressor.Criterion criterion) {
            this.criterion = criterion;
            return this;
        }
        
        public RandomForestRegressor build() {
            return new RandomForestRegressor(this);
        }
    }
    
    /**
     * Default constructor with default hyperparameters.
     */
    public RandomForestRegressor() {
        this(new Builder());
    }
    
    /**
     * Constructor with number of estimators.
     */
    public RandomForestRegressor(int nEstimators) {
        this(new Builder().nEstimators(nEstimators));
    }
    
    @Override
    public void train(double[][] X, double[] y) {
        fit(X, y);
    }
    
    /**
     * Fit the random forest regressor.
     * 
     * @param X Training feature matrix (n_samples x n_features)
     * @param y Training target values (n_samples)
     */
    public void fit(double[][] X, double[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0 || X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same length and be non-empty");
        }
        
        int n = X.length;
        this.numFeatures = X[0].length;
        
        // Calculate max features
        int maxFeaturesInt = calculateMaxFeatures(numFeatures);
        
        // Clear trees if not warm start
        if (!warmStart) {
            trees.clear();
        }
        
        // Initialize OOB tracking
        if (oobScore && bootstrap) {
            oobPredictions = new double[n][1];
            oobCounts = new int[n];
        }
        
        // Train trees
        int numNewTrees = nEstimators - trees.size();
        
        if (nJobs > 1) {
            // Parallel training
            trainParallel(X, y, numNewTrees, maxFeaturesInt);
        } else {
            // Sequential training
            trainSequential(X, y, numNewTrees, maxFeaturesInt);
        }
        
        // Calculate OOB score
        if (oobScore && bootstrap) {
            calculateOobScore(y);
        }
        
        // Aggregate feature importance
        aggregateFeatureImportance();
        
        this.fitted = true;
    }
    
    private void trainSequential(double[][] X, double[] y, int numTrees, int maxFeaturesInt) {
        int n = X.length;
        
        for (int t = 0; t < numTrees; t++) {
            int treeSeed = random.nextInt();
            Random treeRandom = new Random(treeSeed);
            
            // Bootstrap sample
            int[] sampleIndices;
            boolean[] inBag = new boolean[n];
            
            if (bootstrap) {
                sampleIndices = new int[n];
                for (int i = 0; i < n; i++) {
                    int idx = treeRandom.nextInt(n);
                    sampleIndices[i] = idx;
                    inBag[idx] = true;
                }
            } else {
                sampleIndices = IntStream.range(0, n).toArray();
                Arrays.fill(inBag, true);
            }
            
            // Create bootstrap sample
            double[][] XSample = new double[n][];
            double[] ySample = new double[n];
            for (int i = 0; i < n; i++) {
                XSample[i] = X[sampleIndices[i]];
                ySample[i] = y[sampleIndices[i]];
            }
            
            // Train tree
            DecisionTreeRegressor tree = new DecisionTreeRegressor(
                maxDepth, minSamplesSplit, minSamplesLeaf, maxFeaturesInt, treeSeed);
            tree.fit(XSample, ySample);
            trees.add(tree);
            
            // Update OOB predictions
            if (oobScore && bootstrap) {
                for (int i = 0; i < n; i++) {
                    if (!inBag[i]) {
                        double pred = tree.predict(X[i]);
                        oobPredictions[i][0] += pred;
                        oobCounts[i]++;
                    }
                }
            }
        }
    }
    
    private void trainParallel(double[][] X, double[] y, int numTrees, int maxFeaturesInt) {
        int n = X.length;
        int actualJobs = Math.min(nJobs, numTrees);
        executor = Executors.newFixedThreadPool(actualJobs);
        
        // Prepare seeds for reproducibility
        int[] seeds = new int[numTrees];
        for (int i = 0; i < numTrees; i++) {
            seeds[i] = random.nextInt();
        }
        
        List<Future<TreeResult>> futures = new ArrayList<>();
        
        for (int t = 0; t < numTrees; t++) {
            final int treeIdx = t;
            futures.add(executor.submit(() -> trainSingleTree(X, y, seeds[treeIdx], maxFeaturesInt)));
        }
        
        // Collect results
        for (Future<TreeResult> future : futures) {
            try {
                TreeResult result = future.get();
                trees.add(result.tree);
                
                if (oobScore && bootstrap) {
                    for (int i = 0; i < n; i++) {
                        if (!result.inBag[i]) {
                            oobPredictions[i][0] += result.oobPreds[i];
                            oobCounts[i]++;
                        }
                    }
                }
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Error training tree: " + e.getMessage(), e);
            }
        }
        
        executor.shutdown();
    }
    
    private static class TreeResult {
        DecisionTreeRegressor tree;
        boolean[] inBag;
        double[] oobPreds;
    }
    
    private TreeResult trainSingleTree(double[][] X, double[] y, int seed, int maxFeaturesInt) {
        int n = X.length;
        Random treeRandom = new Random(seed);
        
        int[] sampleIndices = new int[n];
        boolean[] inBag = new boolean[n];
        
        if (bootstrap) {
            for (int i = 0; i < n; i++) {
                int idx = treeRandom.nextInt(n);
                sampleIndices[i] = idx;
                inBag[idx] = true;
            }
        } else {
            for (int i = 0; i < n; i++) {
                sampleIndices[i] = i;
                inBag[i] = true;
            }
        }
        
        double[][] XSample = new double[n][];
        double[] ySample = new double[n];
        for (int i = 0; i < n; i++) {
            XSample[i] = X[sampleIndices[i]];
            ySample[i] = y[sampleIndices[i]];
        }
        
        DecisionTreeRegressor tree = new DecisionTreeRegressor(
            maxDepth, minSamplesSplit, minSamplesLeaf, maxFeaturesInt, seed);
        tree.fit(XSample, ySample);
        
        TreeResult result = new TreeResult();
        result.tree = tree;
        result.inBag = inBag;
        
        if (oobScore && bootstrap) {
            result.oobPreds = new double[n];
            for (int i = 0; i < n; i++) {
                if (!inBag[i]) {
                    result.oobPreds[i] = tree.predict(X[i]);
                }
            }
        }
        
        return result;
    }
    
    private int calculateMaxFeatures(int nFeatures) {
        switch (maxFeaturesStrategy) {
            case SQRT:
                return Math.max(1, (int) Math.sqrt(nFeatures));
            case LOG2:
                return Math.max(1, (int) (Math.log(nFeatures) / Math.log(2)));
            case FRACTION:
                return Math.max(1, (int) (nFeatures * maxFeaturesFraction));
            case ALL:
            default:
                return nFeatures;
        }
    }
    
    private void calculateOobScore(double[] y) {
        int n = y.length;
        double sumSE = 0;
        int count = 0;
        
        for (int i = 0; i < n; i++) {
            if (oobCounts[i] > 0) {
                double pred = oobPredictions[i][0] / oobCounts[i];
                double error = y[i] - pred;
                sumSE += error * error;
                count++;
            }
        }
        
        // Calculate R^2 score
        if (count > 0) {
            double mse = sumSE / count;
            double yMean = Arrays.stream(y).average().orElse(0);
            double ssTotal = 0;
            for (int i = 0; i < n; i++) {
                if (oobCounts[i] > 0) {
                    double diff = y[i] - yMean;
                    ssTotal += diff * diff;
                }
            }
            
            if (ssTotal > 0) {
                oobScoreValue = 1 - (sumSE / ssTotal);
            } else {
                oobScoreValue = 0;
            }
        }
    }
    
    private void aggregateFeatureImportance() {
        featureImportance = new double[numFeatures];
        
        for (DecisionTreeRegressor tree : trees) {
            double[] treeImportance = tree.getFeatureImportance();
            if (treeImportance != null) {
                for (int i = 0; i < numFeatures; i++) {
                    featureImportance[i] += treeImportance[i];
                }
            }
        }
        
        // Normalize
        double total = Arrays.stream(featureImportance).sum();
        if (total > 0) {
            for (int i = 0; i < numFeatures; i++) {
                featureImportance[i] /= total;
            }
        }
    }
    
    @Override
    public double predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException("Input must have " + numFeatures + " features");
        }
        
        double sum = 0;
        for (DecisionTreeRegressor tree : trees) {
            sum += tree.predict(x);
        }
        return sum / trees.size();
    }
    
    /**
     * Predicts values for multiple inputs.
     * 
     * @param X array of input features
     * @return array of predicted values
     */
    public double[] predict(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        double[] predictions = new double[X.length];
        
        if (nJobs > 1 && X.length > 100) {
            // Parallel prediction
            int actualJobs = Math.min(nJobs, X.length);
            executor = Executors.newFixedThreadPool(actualJobs);
            
            List<Future<double[]>> futures = new ArrayList<>();
            int chunkSize = (X.length + actualJobs - 1) / actualJobs;
            
            for (int i = 0; i < X.length; i += chunkSize) {
                final int start = i;
                final int end = Math.min(i + chunkSize, X.length);
                
                futures.add(executor.submit(() -> {
                    double[] chunk = new double[end - start];
                    for (int j = start; j < end; j++) {
                        chunk[j - start] = predict(X[j]);
                    }
                    return chunk;
                }));
            }
            
            int idx = 0;
            for (Future<double[]> future : futures) {
                try {
                    double[] chunk = future.get();
                    System.arraycopy(chunk, 0, predictions, idx, chunk.length);
                    idx += chunk.length;
                } catch (InterruptedException | ExecutionException e) {
                    throw new RuntimeException("Error in prediction: " + e.getMessage(), e);
                }
            }
            
            executor.shutdown();
        } else {
            // Sequential prediction
            for (int i = 0; i < X.length; i++) {
                predictions[i] = predict(X[i]);
            }
        }
        
        return predictions;
    }
    
    /**
     * Returns individual predictions from all trees.
     * 
     * @param x input features
     * @return array of predictions from each tree
     */
    public double[] predictAllTrees(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        double[] predictions = new double[trees.size()];
        for (int i = 0; i < trees.size(); i++) {
            predictions[i] = trees.get(i).predict(x);
        }
        return predictions;
    }
    
    /**
     * Returns the prediction variance (uncertainty estimate).
     * 
     * @param x input features
     * @return variance of predictions across trees
     */
    public double predictVariance(double[] x) {
        double[] allPreds = predictAllTrees(x);
        double mean = Arrays.stream(allPreds).average().orElse(0);
        double variance = 0;
        for (double pred : allPreds) {
            double diff = pred - mean;
            variance += diff * diff;
        }
        return variance / allPreds.length;
    }
    
    /**
     * Applies trees to samples and returns leaf indices.
     * 
     * @param X input features
     * @return 2D array of leaf indices (samples x trees)
     */
    public int[][] apply(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        int[][] leafIndices = new int[X.length][trees.size()];
        for (int t = 0; t < trees.size(); t++) {
            int[] treeLeaves = trees.get(t).apply(X);
            for (int i = 0; i < X.length; i++) {
                leafIndices[i][t] = treeLeaves[i];
            }
        }
        return leafIndices;
    }
    
    /**
     * Get feature importance scores.
     * 
     * @return Array of feature importance values (normalized to sum to 1)
     */
    public double[] getFeatureImportance() {
        if (!fitted) {
            return null;
        }
        return featureImportance.clone();
    }
    
    /**
     * Get OOB score (R^2).
     * 
     * @return OOB R^2 score, or NaN if not available
     */
    public double getOobScore() {
        if (!fitted || !oobScore || !bootstrap) {
            return Double.NaN;
        }
        return oobScoreValue;
    }
    
    /**
     * Get the number of trees.
     */
    public int getNEstimators() {
        return trees.size();
    }
    
    /**
     * Get the list of trees (for inspection).
     */
    public List<DecisionTreeRegressor> getEstimators() {
        return Collections.unmodifiableList(trees);
    }
    
    /**
     * Checks if the model has been trained.
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Calculate R^2 score on given data.
     */
    public double score(double[][] X, double[] y) {
        double[] predictions = predict(X);
        
        double yMean = Arrays.stream(y).average().orElse(0);
        double ssRes = 0;
        double ssTot = 0;
        
        for (int i = 0; i < y.length; i++) {
            double error = y[i] - predictions[i];
            ssRes += error * error;
            double diff = y[i] - yMean;
            ssTot += diff * diff;
        }
        
        return ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    }
    
    @Override
    public String toString() {
        return String.format(
            "RandomForestRegressor(nEstimators=%d, maxDepth=%s, maxFeatures=%s, bootstrap=%s, fitted=%s)",
            nEstimators, 
            maxDepth == Integer.MAX_VALUE ? "None" : maxDepth,
            maxFeaturesStrategy,
            bootstrap,
            fitted
        );
    }
}
