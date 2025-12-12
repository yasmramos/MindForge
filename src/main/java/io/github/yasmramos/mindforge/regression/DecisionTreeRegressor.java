package io.github.yasmramos.mindforge.regression;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Decision Tree Regressor implementing the CART (Classification and Regression Trees) algorithm.
 * 
 * <p>Decision trees for regression predict continuous values by partitioning the feature
 * space into regions and predicting the mean target value within each region.</p>
 * 
 * <p>Features:</p>
 * <ul>
 *   <li>Mean Squared Error (MSE) and Mean Absolute Error (MAE) criteria</li>
 *   <li>Configurable maximum depth to prevent overfitting</li>
 *   <li>Minimum samples required to split a node</li>
 *   <li>Minimum samples required at leaf nodes</li>
 *   <li>Feature importance calculation</li>
 *   <li>Support for random feature selection (for ensemble methods)</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * double[][] X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
 * double[] y = {1.2, 2.1, 2.9, 4.1, 5.0};
 * 
 * DecisionTreeRegressor tree = new DecisionTreeRegressor.Builder()
 *     .maxDepth(5)
 *     .minSamplesSplit(2)
 *     .criterion(DecisionTreeRegressor.Criterion.MSE)
 *     .build();
 * 
 * tree.fit(X, y);
 * double prediction = tree.predict(new double[]{2.5});
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class DecisionTreeRegressor implements Regressor<double[]> {
    
    /**
     * Splitting criterion for the decision tree.
     */
    public enum Criterion {
        /** Mean Squared Error - minimizes variance */
        MSE,
        /** Mean Absolute Error - more robust to outliers */
        MAE,
        /** Friedman MSE - improved MSE for gradient boosting */
        FRIEDMAN_MSE
    }
    
    /**
     * Internal node representation in the decision tree.
     */
    static class Node {
        // Split information
        int featureIndex;           // Feature used for splitting (-1 for leaf nodes)
        double threshold;            // Threshold value for the split
        
        // Children
        Node left;                   // Left child (values <= threshold)
        Node right;                  // Right child (values > threshold)
        
        // Leaf information
        double predictedValue;       // Predicted value for leaf nodes
        int numSamples;              // Number of samples in this node
        double impurity;             // Impurity value of this node
        double sumTarget;            // Sum of target values (for incremental updates)
        
        /**
         * Creates a leaf node.
         */
        Node(double predictedValue, int numSamples, double impurity, double sumTarget) {
            this.featureIndex = -1;
            this.predictedValue = predictedValue;
            this.numSamples = numSamples;
            this.impurity = impurity;
            this.sumTarget = sumTarget;
        }
        
        /**
         * Creates an internal (split) node.
         */
        Node(int featureIndex, double threshold, int numSamples, double impurity) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.numSamples = numSamples;
            this.impurity = impurity;
        }
        
        boolean isLeaf() {
            return featureIndex == -1;
        }
    }
    
    // Hyperparameters
    private final int maxDepth;
    private final int minSamplesSplit;
    private final int minSamplesLeaf;
    private final Criterion criterion;
    private final Integer maxFeatures;  // null means use all features
    private final Double minImpurityDecrease;
    private final int randomState;
    
    // Model state
    private Node root;
    private int numFeatures;
    private boolean fitted;
    private Random random;
    private double[] featureImportance;
    
    // Training statistics
    private double trainMSE;
    private double trainMAE;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private DecisionTreeRegressor(Builder builder) {
        this.maxDepth = builder.maxDepth;
        this.minSamplesSplit = builder.minSamplesSplit;
        this.minSamplesLeaf = builder.minSamplesLeaf;
        this.criterion = builder.criterion;
        this.maxFeatures = builder.maxFeatures;
        this.minImpurityDecrease = builder.minImpurityDecrease;
        this.randomState = builder.randomState;
        this.fitted = false;
        this.random = new Random(randomState);
    }
    
    /**
     * Builder pattern for creating DecisionTreeRegressor instances.
     */
    public static class Builder {
        private int maxDepth = Integer.MAX_VALUE;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private Criterion criterion = Criterion.MSE;
        private Integer maxFeatures = null;
        private Double minImpurityDecrease = 0.0;
        private int randomState = 42;
        
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
        
        public Builder criterion(Criterion criterion) {
            this.criterion = criterion;
            return this;
        }
        
        public Builder maxFeatures(Integer maxFeatures) {
            if (maxFeatures != null && maxFeatures <= 0) {
                throw new IllegalArgumentException("maxFeatures must be positive");
            }
            this.maxFeatures = maxFeatures;
            return this;
        }
        
        public Builder minImpurityDecrease(double minImpurityDecrease) {
            if (minImpurityDecrease < 0) {
                throw new IllegalArgumentException("minImpurityDecrease must be non-negative");
            }
            this.minImpurityDecrease = minImpurityDecrease;
            return this;
        }
        
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public DecisionTreeRegressor build() {
            return new DecisionTreeRegressor(this);
        }
    }
    
    /**
     * Default constructor with default hyperparameters.
     */
    public DecisionTreeRegressor() {
        this(new Builder());
    }
    
    /**
     * Constructor for ensemble methods that need specific parameters.
     */
    public DecisionTreeRegressor(int maxDepth, int minSamplesSplit, int minSamplesLeaf, 
                                  Integer maxFeatures, int randomState) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.criterion = Criterion.MSE;
        this.maxFeatures = maxFeatures;
        this.minImpurityDecrease = 0.0;
        this.randomState = randomState;
        this.fitted = false;
        this.random = new Random(randomState);
    }
    
    @Override
    public void train(double[][] X, double[] y) {
        fit(X, y);
    }
    
    /**
     * Fit the decision tree regressor.
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
        
        this.numFeatures = X[0].length;
        this.featureImportance = new double[numFeatures];
        
        // Build the tree recursively
        int[] indices = IntStream.range(0, X.length).toArray();
        this.root = buildTree(X, y, indices, 1);
        this.fitted = true;
        
        // Normalize feature importance
        double totalImportance = Arrays.stream(featureImportance).sum();
        if (totalImportance > 0) {
            for (int i = 0; i < featureImportance.length; i++) {
                featureImportance[i] /= totalImportance;
            }
        }
        
        // Compute training metrics
        computeTrainingMetrics(X, y);
    }
    
    /**
     * Fit the tree with sample weights.
     */
    public void fit(double[][] X, double[] y, double[] sampleWeights) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0 || X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same length and be non-empty");
        }
        if (sampleWeights != null && sampleWeights.length != X.length) {
            throw new IllegalArgumentException("sampleWeights must have same length as X");
        }
        
        this.numFeatures = X[0].length;
        this.featureImportance = new double[numFeatures];
        
        int[] indices = IntStream.range(0, X.length).toArray();
        this.root = buildTreeWeighted(X, y, sampleWeights, indices, 1);
        this.fitted = true;
        
        double totalImportance = Arrays.stream(featureImportance).sum();
        if (totalImportance > 0) {
            for (int i = 0; i < featureImportance.length; i++) {
                featureImportance[i] /= totalImportance;
            }
        }
        
        computeTrainingMetrics(X, y);
    }
    
    private void computeTrainingMetrics(double[][] X, double[] y) {
        double sumSE = 0;
        double sumAE = 0;
        for (int i = 0; i < X.length; i++) {
            double pred = predict(X[i]);
            double error = y[i] - pred;
            sumSE += error * error;
            sumAE += Math.abs(error);
        }
        trainMSE = sumSE / X.length;
        trainMAE = sumAE / X.length;
    }
    
    /**
     * Recursively builds the decision tree.
     */
    private Node buildTree(double[][] X, double[] y, int[] indices, int depth) {
        int numSamples = indices.length;
        
        // Calculate statistics
        double sum = 0;
        double sumSq = 0;
        for (int idx : indices) {
            sum += y[idx];
            sumSq += y[idx] * y[idx];
        }
        double mean = sum / numSamples;
        double impurity = calculateImpurity(y, indices, mean);
        
        // Check stopping criteria
        if (depth >= maxDepth || 
            numSamples < minSamplesSplit || 
            impurity < 1e-10) {
            return new Node(mean, numSamples, impurity, sum);
        }
        
        // Find best split
        Split bestSplit = findBestSplit(X, y, indices, impurity);
        
        if (bestSplit == null || 
            bestSplit.leftIndices.length < minSamplesLeaf || 
            bestSplit.rightIndices.length < minSamplesLeaf ||
            bestSplit.gain < minImpurityDecrease) {
            return new Node(mean, numSamples, impurity, sum);
        }
        
        // Create internal node and recursively build children
        Node node = new Node(bestSplit.featureIndex, bestSplit.threshold, numSamples, impurity);
        node.left = buildTree(X, y, bestSplit.leftIndices, depth + 1);
        node.right = buildTree(X, y, bestSplit.rightIndices, depth + 1);
        
        return node;
    }
    
    /**
     * Recursively builds the decision tree with sample weights.
     */
    private Node buildTreeWeighted(double[][] X, double[] y, double[] weights, 
                                    int[] indices, int depth) {
        int numSamples = indices.length;
        
        // Calculate weighted statistics
        double sumW = 0;
        double sumWY = 0;
        for (int idx : indices) {
            double w = weights != null ? weights[idx] : 1.0;
            sumW += w;
            sumWY += w * y[idx];
        }
        double mean = sumW > 0 ? sumWY / sumW : 0;
        double impurity = calculateWeightedImpurity(y, weights, indices, mean);
        
        // Check stopping criteria
        if (depth >= maxDepth || 
            numSamples < minSamplesSplit || 
            impurity < 1e-10) {
            return new Node(mean, numSamples, impurity, sumWY);
        }
        
        // Find best split
        Split bestSplit = findBestSplitWeighted(X, y, weights, indices, impurity);
        
        if (bestSplit == null || 
            bestSplit.leftIndices.length < minSamplesLeaf || 
            bestSplit.rightIndices.length < minSamplesLeaf ||
            bestSplit.gain < minImpurityDecrease) {
            return new Node(mean, numSamples, impurity, sumWY);
        }
        
        // Create internal node and recursively build children
        Node node = new Node(bestSplit.featureIndex, bestSplit.threshold, numSamples, impurity);
        node.left = buildTreeWeighted(X, y, weights, bestSplit.leftIndices, depth + 1);
        node.right = buildTreeWeighted(X, y, weights, bestSplit.rightIndices, depth + 1);
        
        return node;
    }
    
    /**
     * Helper class to store split information.
     */
    private static class Split {
        int featureIndex;
        double threshold;
        int[] leftIndices;
        int[] rightIndices;
        double gain;
        
        Split(int featureIndex, double threshold, int[] leftIndices, int[] rightIndices, double gain) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.leftIndices = leftIndices;
            this.rightIndices = rightIndices;
            this.gain = gain;
        }
    }
    
    /**
     * Finds the best split for the current node.
     */
    private Split findBestSplit(double[][] X, double[] y, int[] indices, double parentImpurity) {
        Split bestSplit = null;
        double bestGain = Double.NEGATIVE_INFINITY;
        
        // Select features to consider
        int[] featuresToTry = selectFeatures();
        
        // Try selected features
        for (int featureIdx : featuresToTry) {
            // Get sorted values and their indices
            double[][] valueIndexPairs = new double[indices.length][2];
            for (int i = 0; i < indices.length; i++) {
                valueIndexPairs[i][0] = X[indices[i]][featureIdx];
                valueIndexPairs[i][1] = indices[i];
            }
            Arrays.sort(valueIndexPairs, Comparator.comparingDouble(a -> a[0]));
            
            // Compute cumulative sums for efficient impurity calculation
            double leftSum = 0;
            double leftSumSq = 0;
            double rightSum = 0;
            double rightSumSq = 0;
            
            for (int i = 0; i < indices.length; i++) {
                int idx = (int) valueIndexPairs[i][1];
                rightSum += y[idx];
                rightSumSq += y[idx] * y[idx];
            }
            
            // Try all possible thresholds
            for (int i = 0; i < indices.length - 1; i++) {
                int idx = (int) valueIndexPairs[i][1];
                double val = y[idx];
                
                leftSum += val;
                leftSumSq += val * val;
                rightSum -= val;
                rightSumSq -= val * val;
                
                int leftCount = i + 1;
                int rightCount = indices.length - leftCount;
                
                // Skip if same value as next
                if (valueIndexPairs[i][0] == valueIndexPairs[i + 1][0]) {
                    continue;
                }
                
                // Skip if would violate minSamplesLeaf
                if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) {
                    continue;
                }
                
                double threshold = (valueIndexPairs[i][0] + valueIndexPairs[i + 1][0]) / 2.0;
                
                // Calculate impurity reduction
                double leftMean = leftSum / leftCount;
                double rightMean = rightSum / rightCount;
                
                double leftImpurity = leftSumSq / leftCount - leftMean * leftMean;
                double rightImpurity = rightSumSq / rightCount - rightMean * rightMean;
                
                double weightedChildImpurity = (leftCount * leftImpurity + rightCount * rightImpurity) / indices.length;
                double gain = parentImpurity - weightedChildImpurity;
                
                // For Friedman MSE, add improvement term
                if (criterion == Criterion.FRIEDMAN_MSE) {
                    double diff = leftMean - rightMean;
                    gain = (leftCount * rightCount) * diff * diff / (indices.length * indices.length);
                }
                
                if (gain > bestGain) {
                    bestGain = gain;
                    
                    // Build index arrays
                    int[] leftIndices = new int[leftCount];
                    int[] rightIndices = new int[rightCount];
                    for (int j = 0; j <= i; j++) {
                        leftIndices[j] = (int) valueIndexPairs[j][1];
                    }
                    for (int j = i + 1; j < indices.length; j++) {
                        rightIndices[j - i - 1] = (int) valueIndexPairs[j][1];
                    }
                    
                    bestSplit = new Split(featureIdx, threshold, leftIndices, rightIndices, gain);
                }
            }
        }
        
        // Update feature importance
        if (bestSplit != null) {
            double importance = bestSplit.gain * indices.length;
            featureImportance[bestSplit.featureIndex] += importance;
        }
        
        return bestSplit;
    }
    
    /**
     * Finds the best split with sample weights.
     */
    private Split findBestSplitWeighted(double[][] X, double[] y, double[] weights,
                                         int[] indices, double parentImpurity) {
        Split bestSplit = null;
        double bestGain = Double.NEGATIVE_INFINITY;
        
        int[] featuresToTry = selectFeatures();
        
        for (int featureIdx : featuresToTry) {
            double[][] valueIndexPairs = new double[indices.length][2];
            for (int i = 0; i < indices.length; i++) {
                valueIndexPairs[i][0] = X[indices[i]][featureIdx];
                valueIndexPairs[i][1] = indices[i];
            }
            Arrays.sort(valueIndexPairs, Comparator.comparingDouble(a -> a[0]));
            
            double leftSumW = 0, leftSumWY = 0, leftSumWYY = 0;
            double rightSumW = 0, rightSumWY = 0, rightSumWYY = 0;
            
            for (int i = 0; i < indices.length; i++) {
                int idx = (int) valueIndexPairs[i][1];
                double w = weights != null ? weights[idx] : 1.0;
                rightSumW += w;
                rightSumWY += w * y[idx];
                rightSumWYY += w * y[idx] * y[idx];
            }
            
            for (int i = 0; i < indices.length - 1; i++) {
                int idx = (int) valueIndexPairs[i][1];
                double w = weights != null ? weights[idx] : 1.0;
                double val = y[idx];
                
                leftSumW += w;
                leftSumWY += w * val;
                leftSumWYY += w * val * val;
                rightSumW -= w;
                rightSumWY -= w * val;
                rightSumWYY -= w * val * val;
                
                if (valueIndexPairs[i][0] == valueIndexPairs[i + 1][0]) {
                    continue;
                }
                
                int leftCount = i + 1;
                int rightCount = indices.length - leftCount;
                
                if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) {
                    continue;
                }
                
                if (leftSumW < 1e-10 || rightSumW < 1e-10) {
                    continue;
                }
                
                double threshold = (valueIndexPairs[i][0] + valueIndexPairs[i + 1][0]) / 2.0;
                
                double leftMean = leftSumWY / leftSumW;
                double rightMean = rightSumWY / rightSumW;
                
                double leftImpurity = leftSumWYY / leftSumW - leftMean * leftMean;
                double rightImpurity = rightSumWYY / rightSumW - rightMean * rightMean;
                
                double totalW = leftSumW + rightSumW;
                double weightedChildImpurity = (leftSumW * leftImpurity + rightSumW * rightImpurity) / totalW;
                double gain = parentImpurity - weightedChildImpurity;
                
                if (gain > bestGain) {
                    bestGain = gain;
                    
                    int[] leftIndices = new int[leftCount];
                    int[] rightIndices = new int[rightCount];
                    for (int j = 0; j <= i; j++) {
                        leftIndices[j] = (int) valueIndexPairs[j][1];
                    }
                    for (int j = i + 1; j < indices.length; j++) {
                        rightIndices[j - i - 1] = (int) valueIndexPairs[j][1];
                    }
                    
                    bestSplit = new Split(featureIdx, threshold, leftIndices, rightIndices, gain);
                }
            }
        }
        
        if (bestSplit != null) {
            featureImportance[bestSplit.featureIndex] += bestSplit.gain * indices.length;
        }
        
        return bestSplit;
    }
    
    private int[] selectFeatures() {
        if (maxFeatures != null && maxFeatures < numFeatures) {
            return random.ints(0, numFeatures)
                         .distinct()
                         .limit(maxFeatures)
                         .toArray();
        } else {
            return IntStream.range(0, numFeatures).toArray();
        }
    }
    
    /**
     * Calculates impurity based on the chosen criterion.
     */
    private double calculateImpurity(double[] y, int[] indices, double mean) {
        if (criterion == Criterion.MAE) {
            double sumAE = 0;
            for (int idx : indices) {
                sumAE += Math.abs(y[idx] - mean);
            }
            return sumAE / indices.length;
        } else {
            // MSE or FRIEDMAN_MSE
            double sumSE = 0;
            for (int idx : indices) {
                double diff = y[idx] - mean;
                sumSE += diff * diff;
            }
            return sumSE / indices.length;
        }
    }
    
    private double calculateWeightedImpurity(double[] y, double[] weights, int[] indices, double mean) {
        double sumW = 0;
        double sumWE = 0;
        
        for (int idx : indices) {
            double w = weights != null ? weights[idx] : 1.0;
            sumW += w;
            if (criterion == Criterion.MAE) {
                sumWE += w * Math.abs(y[idx] - mean);
            } else {
                double diff = y[idx] - mean;
                sumWE += w * diff * diff;
            }
        }
        
        return sumW > 0 ? sumWE / sumW : 0;
    }
    
    @Override
    public double predict(double[] x) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        if (x.length != numFeatures) {
            throw new IllegalArgumentException("Input must have " + numFeatures + " features");
        }
        
        Node node = root;
        while (!node.isLeaf()) {
            if (x[node.featureIndex] <= node.threshold) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        
        return node.predictedValue;
    }
    
    /**
     * Predicts values for multiple inputs.
     * 
     * @param X array of input features
     * @return array of predicted values
     */
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Applies the tree to samples and returns leaf indices.
     * 
     * @param X input features
     * @return array of leaf node indices
     */
    public int[] apply(double[][] X) {
        if (!fitted) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        int[] leafIndices = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            leafIndices[i] = getLeafIndex(X[i], root, 0);
        }
        return leafIndices;
    }
    
    private int getLeafIndex(double[] x, Node node, int currentIdx) {
        if (node.isLeaf()) {
            return currentIdx;
        }
        if (x[node.featureIndex] <= node.threshold) {
            return getLeafIndex(x, node.left, 2 * currentIdx + 1);
        } else {
            return getLeafIndex(x, node.right, 2 * currentIdx + 2);
        }
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
     * Returns the actual depth of the trained tree.
     */
    public int getTreeDepth() {
        if (!fitted) {
            return 0;
        }
        return calculateDepth(root);
    }
    
    private int calculateDepth(Node node) {
        if (node.isLeaf()) {
            return 1;
        }
        return 1 + Math.max(calculateDepth(node.left), calculateDepth(node.right));
    }
    
    /**
     * Returns the number of leaf nodes in the tree.
     */
    public int getNumLeaves() {
        if (!fitted) {
            return 0;
        }
        return countLeaves(root);
    }
    
    private int countLeaves(Node node) {
        if (node.isLeaf()) {
            return 1;
        }
        return countLeaves(node.left) + countLeaves(node.right);
    }
    
    /**
     * Returns the total number of nodes in the tree.
     */
    public int getNumNodes() {
        if (!fitted) {
            return 0;
        }
        return countNodes(root);
    }
    
    private int countNodes(Node node) {
        if (node.isLeaf()) {
            return 1;
        }
        return 1 + countNodes(node.left) + countNodes(node.right);
    }
    
    /**
     * Get training MSE.
     */
    public double getTrainMSE() {
        return trainMSE;
    }
    
    /**
     * Get training MAE.
     */
    public double getTrainMAE() {
        return trainMAE;
    }
    
    /**
     * Checks if the model has been trained.
     */
    public boolean isFitted() {
        return fitted;
    }
    
    /**
     * Get the root node (for ensemble methods).
     */
    Node getRoot() {
        return root;
    }
    
    /**
     * Get maximum depth setting.
     */
    public int getMaxDepth() {
        return maxDepth;
    }
    
    /**
     * Get criterion.
     */
    public Criterion getCriterion() {
        return criterion;
    }
    
    /**
     * Returns information about the tree structure as a string.
     */
    public String getTreeInfo() {
        if (!fitted) {
            return "Tree not fitted yet";
        }
        return String.format("DecisionTreeRegressor(depth=%d, leaves=%d, nodes=%d, trainMSE=%.4f)", 
                           getTreeDepth(), getNumLeaves(), getNumNodes(), trainMSE);
    }
    
    @Override
    public String toString() {
        return String.format("DecisionTreeRegressor(maxDepth=%d, minSamplesSplit=%d, minSamplesLeaf=%d, criterion=%s, fitted=%s)",
                           maxDepth, minSamplesSplit, minSamplesLeaf, criterion, fitted);
    }
}
