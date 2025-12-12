package io.github.yasmramos.mindforge.classification;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Decision Tree Classifier implementing the CART (Classification and Regression Trees) algorithm.
 * 
 * <p>Decision trees are non-parametric supervised learning methods used for classification.
 * The tree is built by recursively splitting the data based on features that provide
 * the best separation according to a criterion (Gini impurity or Entropy).</p>
 * 
 * <p>Features:</p>
 * <ul>
 *   <li>Support for Gini impurity and Entropy criteria</li>
 *   <li>Configurable maximum depth to prevent overfitting</li>
 *   <li>Minimum samples required to split a node</li>
 *   <li>Minimum samples required at leaf nodes</li>
 *   <li>Probability predictions for each class</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>{@code
 * double[][] X = {{2.0, 3.0}, {1.0, 1.0}, {3.0, 4.0}, {5.0, 6.0}};
 * int[] y = {0, 0, 1, 1};
 * 
 * DecisionTreeClassifier tree = new DecisionTreeClassifier.Builder()
 *     .maxDepth(5)
 *     .minSamplesSplit(2)
 *     .criterion(DecisionTreeClassifier.Criterion.GINI)
 *     .build();
 * 
 * tree.train(X, y);
 * int prediction = tree.predict(new double[]{2.5, 3.5});
 * double[] probabilities = tree.predictProba(new double[]{2.5, 3.5});
 * }</pre>
 */
public class DecisionTreeClassifier implements Classifier<double[]> {
    
    /**
     * Splitting criterion for the decision tree.
     */
    public enum Criterion {
        /** Gini impurity - measures the probability of incorrect classification */
        GINI,
        /** Entropy - measures the information gain from a split */
        ENTROPY
    }
    
    /**
     * Internal node representation in the decision tree.
     */
    private static class Node {
        // Split information
        int featureIndex;           // Feature used for splitting (-1 for leaf nodes)
        double threshold;            // Threshold value for the split
        
        // Children
        Node left;                   // Left child (values <= threshold)
        Node right;                  // Right child (values > threshold)
        
        // Leaf information
        int predictedClass;          // Predicted class for leaf nodes
        double[] classProbabilities; // Probability distribution over classes
        int numSamples;              // Number of samples in this node
        double impurity;             // Impurity value of this node
        
        /**
         * Creates a leaf node.
         */
        Node(int predictedClass, double[] classProbabilities, int numSamples, double impurity) {
            this.featureIndex = -1;
            this.predictedClass = predictedClass;
            this.classProbabilities = classProbabilities;
            this.numSamples = numSamples;
            this.impurity = impurity;
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
    private final int randomState;
    
    // Model state
    private Node root;
    private int numClasses;
    private int numFeatures;
    private boolean fitted;
    private Random random;
    private double[] featureImportance;
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private DecisionTreeClassifier(Builder builder) {
        this.maxDepth = builder.maxDepth;
        this.minSamplesSplit = builder.minSamplesSplit;
        this.minSamplesLeaf = builder.minSamplesLeaf;
        this.criterion = builder.criterion;
        this.maxFeatures = builder.maxFeatures;
        this.randomState = builder.randomState;
        this.fitted = false;
        this.random = new Random(randomState);
    }
    
    /**
     * Builder pattern for creating DecisionTreeClassifier instances.
     */
    public static class Builder {
        private int maxDepth = Integer.MAX_VALUE;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private Criterion criterion = Criterion.GINI;
        private Integer maxFeatures = null;  // null means use all features
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
        
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public DecisionTreeClassifier build() {
            return new DecisionTreeClassifier(this);
        }
    }
    
    /**
     * Default constructor with default hyperparameters.
     */
    public DecisionTreeClassifier() {
        this(new Builder());
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        fit(X, y);
    }
    
    /**
     * Fit the decision tree classifier.
     * 
     * @param X Training feature matrix (n_samples × n_features)
     * @param y Training labels (n_samples)
     */
    public void fit(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("Training data cannot be null");
        }
        if (X.length == 0 || X.length != y.length) {
            throw new IllegalArgumentException("X and y must have the same length and be non-empty");
        }
        
        this.numFeatures = X[0].length;
        this.numClasses = Arrays.stream(y).max().orElse(0) + 1;
        this.featureImportance = new double[numFeatures];
        
        // Build the tree recursively (start at depth 1 for root)
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
    }
    
    /**
     * Recursively builds the decision tree.
     * 
     * @param X feature matrix
     * @param y labels
     * @param indices indices of samples in current node
     * @param depth current depth in the tree
     * @return root node of the subtree
     */
    private Node buildTree(double[][] X, int[] y, int[] indices, int depth) {
        int numSamples = indices.length;
        
        // Calculate class probabilities and impurity
        double[] classCounts = new double[numClasses];
        for (int idx : indices) {
            classCounts[y[idx]]++;
        }
        
        double[] classProbabilities = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            classProbabilities[i] = classCounts[i] / numSamples;
        }
        
        double impurity = calculateImpurity(classProbabilities);
        int predictedClass = argmax(classCounts);
        
        // Check stopping criteria
        // depth starts at 1 for root, maxDepth=1 means only root (no splits)
        if (depth >= maxDepth || 
            numSamples < minSamplesSplit || 
            impurity == 0.0 ||
            numClasses == 1) {
            return new Node(predictedClass, classProbabilities, numSamples, impurity);
        }
        
        // Find best split
        Split bestSplit = findBestSplit(X, y, indices);
        
        if (bestSplit == null || 
            bestSplit.leftIndices.length < minSamplesLeaf || 
            bestSplit.rightIndices.length < minSamplesLeaf) {
            return new Node(predictedClass, classProbabilities, numSamples, impurity);
        }
        
        // Create internal node and recursively build children
        Node node = new Node(bestSplit.featureIndex, bestSplit.threshold, numSamples, impurity);
        node.left = buildTree(X, y, bestSplit.leftIndices, depth + 1);
        node.right = buildTree(X, y, bestSplit.rightIndices, depth + 1);
        
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
    private Split findBestSplit(double[][] X, int[] y, int[] indices) {
        Split bestSplit = null;
        double bestGain = Double.NEGATIVE_INFINITY;
        
        // Calculate parent impurity
        double[] parentClassCounts = new double[numClasses];
        for (int idx : indices) {
            parentClassCounts[y[idx]]++;
        }
        double[] parentProbs = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            parentProbs[i] = parentClassCounts[i] / indices.length;
        }
        double parentImpurity = calculateImpurity(parentProbs);
        
        
        // Select features to consider
        int[] featuresToTry;
        if (maxFeatures != null && maxFeatures < numFeatures) {
            // Random feature selection
            featuresToTry = random.ints(0, numFeatures)
                                  .distinct()
                                  .limit(maxFeatures)
                                  .toArray();
        } else {
            // Use all features
            featuresToTry = IntStream.range(0, numFeatures).toArray();
        }
        
        // Try selected features
        for (int featureIdx : featuresToTry) {
            // Get unique values for this feature
            Set<Double> uniqueValues = new TreeSet<>();
            for (int idx : indices) {
                uniqueValues.add(X[idx][featureIdx]);
            }
            
            // Try all possible thresholds (midpoints between consecutive unique values)
            List<Double> sortedValues = new ArrayList<>(uniqueValues);
            for (int i = 0; i < sortedValues.size() - 1; i++) {
                double threshold = (sortedValues.get(i) + sortedValues.get(i + 1)) / 2.0;
                
                // Split the data
                List<Integer> leftList = new ArrayList<>();
                List<Integer> rightList = new ArrayList<>();
                
                for (int idx : indices) {
                    if (X[idx][featureIdx] <= threshold) {
                        leftList.add(idx);
                    } else {
                        rightList.add(idx);
                    }
                }
                
                if (leftList.isEmpty() || rightList.isEmpty()) {
                    continue;
                }
                
                // Calculate information gain
                int[] leftIndices = leftList.stream().mapToInt(Integer::intValue).toArray();
                int[] rightIndices = rightList.stream().mapToInt(Integer::intValue).toArray();
                
                double gain = calculateGain(y, indices, leftIndices, rightIndices, parentImpurity);
                
                if (gain > bestGain) {
                    bestGain = gain;
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
     * Calculates the information gain from a split.
     */
    private double calculateGain(int[] y, int[] parentIndices, int[] leftIndices, int[] rightIndices, double parentImpurity) {
        int nParent = parentIndices.length;
        int nLeft = leftIndices.length;
        int nRight = rightIndices.length;
        
        // Calculate left impurity
        double[] leftClassCounts = new double[numClasses];
        for (int idx : leftIndices) {
            leftClassCounts[y[idx]]++;
        }
        double[] leftProbs = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            leftProbs[i] = leftClassCounts[i] / nLeft;
        }
        double leftImpurity = calculateImpurity(leftProbs);
        
        // Calculate right impurity
        double[] rightClassCounts = new double[numClasses];
        for (int idx : rightIndices) {
            rightClassCounts[y[idx]]++;
        }
        double[] rightProbs = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            rightProbs[i] = rightClassCounts[i] / nRight;
        }
        double rightImpurity = calculateImpurity(rightProbs);
        
        // Information gain = parent impurity - weighted average of children impurities
        double weightedChildImpurity = (nLeft * leftImpurity + nRight * rightImpurity) / nParent;
        return parentImpurity - weightedChildImpurity;
    }
    
    /**
     * Calculates impurity based on the chosen criterion.
     */
    private double calculateImpurity(double[] probabilities) {
        if (criterion == Criterion.GINI) {
            return calculateGini(probabilities);
        } else {
            return calculateEntropy(probabilities);
        }
    }
    
    /**
     * Calculates Gini impurity.
     * Gini = 1 - sum(p_i^2)
     */
    private double calculateGini(double[] probabilities) {
        double gini = 1.0;
        for (double p : probabilities) {
            gini -= p * p;
        }
        return gini;
    }
    
    /**
     * Calculates entropy.
     * Entropy = -sum(p_i * log2(p_i))
     */
    private double calculateEntropy(double[] probabilities) {
        double entropy = 0.0;
        for (double p : probabilities) {
            if (p > 0) {
                entropy -= p * Math.log(p) / Math.log(2);
            }
        }
        return entropy;
    }
    
    /**
     * Returns the index of the maximum value in the array.
     */
    private int argmax(double[] values) {
        int maxIdx = 0;
        double maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) {
                maxVal = values[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    @Override
    public int predict(double[] x) {
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
        
        return node.predictedClass;
    }
    
    /**
     * Predicts class labels for multiple inputs.
     * 
     * @param X array of input features
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
     * Predicts class probabilities for a single input.
     * 
     * @param x input features
     * @return array of probabilities for each class
     */
    public double[] predictProba(double[] x) {
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
        
        return node.classProbabilities.clone();
    }
    
    /**
     * Predicts class probabilities for multiple inputs.
     * 
     * @param X array of input features
     * @return 2D array where each row contains probabilities for each class
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
     * Returns the maximum depth of the tree.
     */
    public int getMaxDepth() {
        return maxDepth;
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
     * Checks if the model has been trained.
     */
    public boolean isFitted() {
        return fitted;
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
     * Returns information about the tree structure as a string.
     */
    public String getTreeInfo() {
        if (!fitted) {
            return "Tree not fitted yet";
        }
        return String.format("DecisionTree(depth=%d, leaves=%d, samples=%d)", 
                           getTreeDepth(), getNumLeaves(), root.numSamples);
    }
    
    @Override
    public String toString() {
        return String.format("DecisionTreeClassifier(maxDepth=%d, minSamplesSplit=%d, minSamplesLeaf=%d, criterion=%s, fitted=%s)",
                           maxDepth, minSamplesSplit, minSamplesLeaf, criterion, fitted);
    }
}
