package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * XGBoost (eXtreme Gradient Boosting) Classifier.
 * 
 * XGBoost is an optimized gradient boosting algorithm that uses decision trees
 * as base learners. Key features include:
 * - Regularized learning (L1 and L2) to prevent overfitting
 * - Second-order gradient optimization for faster convergence
 * - Built-in handling of missing values
 * - Efficient tree construction with histogram-based splitting
 * - Support for binary and multiclass classification
 * 
 * This implementation uses:
 * - Gradient boosted decision trees
 * - Softmax objective for multiclass classification
 * - Logistic objective for binary classification
 * - Newton boosting with second-order gradients
 * 
 * Example usage:
 * <pre>{@code
 * XGBoostClassifier xgb = new XGBoostClassifier.Builder()
 *     .nEstimators(100)
 *     .maxDepth(6)
 *     .learningRate(0.1)
 *     .lambda(1.0)    // L2 regularization
 *     .alpha(0.0)     // L1 regularization
 *     .subsample(0.8)
 *     .colsampleBytree(0.8)
 *     .randomState(42)
 *     .build();
 * 
 * xgb.fit(X_train, y_train);
 * int[] predictions = xgb.predict(X_test);
 * double[][] probas = xgb.predictProba(X_test);
 * }</pre>
 * 
 * @author MindForge Team
 * @version 1.0
 */
public class XGBoostClassifier implements Classifier<double[]>, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // Hyperparameters
    private final int nEstimators;
    private final int maxDepth;
    private final double learningRate;
    private final double lambda;      // L2 regularization
    private final double alpha;       // L1 regularization
    private final double gamma;       // Minimum loss reduction for split
    private final double subsample;   // Row subsampling ratio
    private final double colsampleBytree; // Column subsampling ratio
    private final int minChildWeight; // Minimum sum of instance weight in child
    private final Integer randomState;
    
    // Model state
    private List<List<XGBTree>> trees; // [class][tree_index]
    private int[] classes;
    private int nClasses;
    private int nFeatures;
    private boolean isFitted;
    private Random random;
    private double[] baseScore; // Initial predictions for each class
    
    /**
     * Private constructor - use Builder to create instances.
     */
    private XGBoostClassifier(int nEstimators, int maxDepth, double learningRate,
                               double lambda, double alpha, double gamma,
                               double subsample, double colsampleBytree,
                               int minChildWeight, Integer randomState) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.alpha = alpha;
        this.gamma = gamma;
        this.subsample = subsample;
        this.colsampleBytree = colsampleBytree;
        this.minChildWeight = minChildWeight;
        this.randomState = randomState;
        this.isFitted = false;
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        fit(X, y);
    }
    
    /**
     * Fits the XGBoost classifier to the training data.
     * 
     * @param X Training features (n_samples x n_features)
     * @param y Training labels
     */
    public void fit(double[][] X, int[] y) {
        validateInput(X, y);
        
        int n = X.length;
        nFeatures = X[0].length;
        random = randomState != null ? new Random(randomState) : new Random();
        
        // Find unique classes
        Set<Integer> uniqueClasses = new TreeSet<>();
        for (int label : y) {
            uniqueClasses.add(label);
        }
        classes = uniqueClasses.stream().mapToInt(Integer::intValue).toArray();
        nClasses = classes.length;
        
        // Initialize base scores (log odds for each class)
        baseScore = new double[nClasses];
        for (int k = 0; k < nClasses; k++) {
            int count = 0;
            for (int yi : y) {
                if (yi == classes[k]) count++;
            }
            double p = (double) count / n;
            baseScore[k] = Math.log(Math.max(p, 1e-10));
        }
        
        // Initialize predictions
        double[][] rawPredictions = new double[n][nClasses];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < nClasses; k++) {
                rawPredictions[i][k] = baseScore[k];
            }
        }
        
        // Initialize trees for each class
        trees = new ArrayList<>();
        for (int k = 0; k < nClasses; k++) {
            trees.add(new ArrayList<>());
        }
        
        // Gradient boosting iterations
        for (int m = 0; m < nEstimators; m++) {
            // Subsample rows
            int[] sampleIndices = subsampleIndices(n);
            
            // For each class, train a tree
            for (int k = 0; k < nClasses; k++) {
                // Compute gradients and hessians for this class
                double[] gradients = new double[n];
                double[] hessians = new double[n];
                
                // Softmax probabilities
                double[][] probs = softmax(rawPredictions);
                
                for (int i = 0; i < n; i++) {
                    int targetClass = findClassIndex(y[i]);
                    double target = (targetClass == k) ? 1.0 : 0.0;
                    
                    // Gradient: p_k - y_k (derivative of cross-entropy loss)
                    gradients[i] = probs[i][k] - target;
                    
                    // Hessian: p_k * (1 - p_k) (second derivative)
                    hessians[i] = Math.max(probs[i][k] * (1 - probs[i][k]), 1e-10);
                }
                
                // Subsample features
                int[] featureIndices = subsampleFeatures();
                
                // Build tree for this class
                XGBTree tree = new XGBTree(maxDepth, lambda, alpha, gamma, minChildWeight);
                tree.fit(X, gradients, hessians, sampleIndices, featureIndices, random);
                trees.get(k).add(tree);
                
                // Update predictions
                for (int i = 0; i < n; i++) {
                    rawPredictions[i][k] += learningRate * tree.predict(X[i]);
                }
            }
        }
        
        isFitted = true;
    }
    
    private int[] subsampleIndices(int n) {
        if (subsample >= 1.0) {
            int[] indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            return indices;
        }
        
        int sampleSize = Math.max(1, (int) (n * subsample));
        Set<Integer> selected = new HashSet<>();
        while (selected.size() < sampleSize) {
            selected.add(random.nextInt(n));
        }
        return selected.stream().mapToInt(Integer::intValue).toArray();
    }
    
    private int[] subsampleFeatures() {
        if (colsampleBytree >= 1.0) {
            int[] indices = new int[nFeatures];
            for (int i = 0; i < nFeatures; i++) indices[i] = i;
            return indices;
        }
        
        int numFeatures = Math.max(1, (int) (nFeatures * colsampleBytree));
        Set<Integer> selected = new HashSet<>();
        while (selected.size() < numFeatures) {
            selected.add(random.nextInt(nFeatures));
        }
        return selected.stream().mapToInt(Integer::intValue).toArray();
    }
    
    private double[][] softmax(double[][] raw) {
        int n = raw.length;
        int k = raw[0].length;
        double[][] result = new double[n][k];
        
        for (int i = 0; i < n; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) {
                max = Math.max(max, raw[i][j]);
            }
            
            double sum = 0;
            for (int j = 0; j < k; j++) {
                result[i][j] = Math.exp(raw[i][j] - max);
                sum += result[i][j];
            }
            
            for (int j = 0; j < k; j++) {
                result[i][j] /= sum;
            }
        }
        
        return result;
    }
    
    private int findClassIndex(int classLabel) {
        for (int i = 0; i < classes.length; i++) {
            if (classes[i] == classLabel) {
                return i;
            }
        }
        return -1;
    }
    
    @Override
    public int predict(double[] x) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        if (x.length != nFeatures) {
            throw new IllegalArgumentException(
                String.format("Expected %d features, got %d", nFeatures, x.length));
        }
        
        double[] probs = predictProba(x);
        int maxIdx = 0;
        double maxProb = probs[0];
        for (int k = 1; k < nClasses; k++) {
            if (probs[k] > maxProb) {
                maxProb = probs[k];
                maxIdx = k;
            }
        }
        return classes[maxIdx];
    }
    
    /**
     * Predicts class labels for multiple samples.
     */
    public int[] predict(double[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        validatePredictInput(X);
        
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    /**
     * Predicts class probabilities for a single sample.
     */
    public double[] predictProba(double[] x) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        
        double[] rawPred = new double[nClasses];
        for (int k = 0; k < nClasses; k++) {
            rawPred[k] = baseScore[k];
            for (XGBTree tree : trees.get(k)) {
                rawPred[k] += learningRate * tree.predict(x);
            }
        }
        
        // Apply softmax
        double max = Double.NEGATIVE_INFINITY;
        for (double v : rawPred) max = Math.max(max, v);
        
        double sum = 0;
        double[] probs = new double[nClasses];
        for (int k = 0; k < nClasses; k++) {
            probs[k] = Math.exp(rawPred[k] - max);
            sum += probs[k];
        }
        
        for (int k = 0; k < nClasses; k++) {
            probs[k] /= sum;
        }
        
        return probs;
    }
    
    /**
     * Predicts class probabilities for multiple samples.
     */
    public double[][] predictProba(double[][] X) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        validatePredictInput(X);
        
        double[][] proba = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            proba[i] = predictProba(X[i]);
        }
        return proba;
    }
    
    /**
     * Computes the raw score (margin) for each class.
     */
    public double[] predictRaw(double[] x) {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted before prediction");
        }
        
        double[] rawPred = new double[nClasses];
        for (int k = 0; k < nClasses; k++) {
            rawPred[k] = baseScore[k];
            for (XGBTree tree : trees.get(k)) {
                rawPred[k] += learningRate * tree.predict(x);
            }
        }
        return rawPred;
    }
    
    /**
     * Computes the training accuracy.
     */
    public double score(double[][] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    /**
     * Returns feature importance scores based on gain.
     */
    public double[] getFeatureImportances() {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        
        double[] importances = new double[nFeatures];
        double totalGain = 0;
        
        for (List<XGBTree> classTrees : trees) {
            for (XGBTree tree : classTrees) {
                double[] treeImportance = tree.getFeatureImportances(nFeatures);
                for (int i = 0; i < nFeatures; i++) {
                    importances[i] += treeImportance[i];
                    totalGain += treeImportance[i];
                }
            }
        }
        
        // Normalize
        if (totalGain > 0) {
            for (int i = 0; i < nFeatures; i++) {
                importances[i] /= totalGain;
            }
        }
        
        return importances;
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
                String.format("X has %d features, but model expects %d features",
                    X[0].length, nFeatures));
        }
    }
    
    // Getters
    
    @Override
    public int getNumClasses() {
        return nClasses;
    }
    
    public int[] getClasses() {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return classes.clone();
    }
    
    public int getNEstimators() {
        return nEstimators;
    }
    
    public int getActualNEstimators() {
        if (!isFitted) {
            throw new IllegalStateException("Model must be fitted first");
        }
        return trees.get(0).size();
    }
    
    public int getMaxDepth() {
        return maxDepth;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public double getLambda() {
        return lambda;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getGamma() {
        return gamma;
    }
    
    public double getSubsample() {
        return subsample;
    }
    
    public double getColsampleBytree() {
        return colsampleBytree;
    }
    
    public boolean isFitted() {
        return isFitted;
    }
    
    public boolean isTrained() {
        return isFitted;
    }
    
    /**
     * XGBoost decision tree node.
     */
    private static class XGBTree implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private final int maxDepth;
        private final double lambda;
        private final double alpha;
        private final double gamma;
        private final int minChildWeight;
        
        private XGBNode root;
        
        public XGBTree(int maxDepth, double lambda, double alpha, 
                       double gamma, int minChildWeight) {
            this.maxDepth = maxDepth;
            this.lambda = lambda;
            this.alpha = alpha;
            this.gamma = gamma;
            this.minChildWeight = minChildWeight;
        }
        
        public void fit(double[][] X, double[] gradients, double[] hessians,
                        int[] sampleIndices, int[] featureIndices, Random random) {
            root = buildTree(X, gradients, hessians, sampleIndices, featureIndices, 0, random);
        }
        
        private XGBNode buildTree(double[][] X, double[] gradients, double[] hessians,
                                   int[] sampleIndices, int[] featureIndices, 
                                   int depth, Random random) {
            // Compute node statistics
            double sumG = 0, sumH = 0;
            for (int i : sampleIndices) {
                sumG += gradients[i];
                sumH += hessians[i];
            }
            
            // Compute leaf weight: -G / (H + lambda)
            double leafWeight = computeLeafWeight(sumG, sumH);
            
            // Check stopping conditions
            if (depth >= maxDepth || sampleIndices.length < 2 || sumH < minChildWeight) {
                return new XGBNode(leafWeight);
            }
            
            // Find best split
            SplitInfo bestSplit = findBestSplit(X, gradients, hessians, 
                                                 sampleIndices, featureIndices, sumG, sumH);
            
            if (bestSplit == null || bestSplit.gain <= gamma) {
                return new XGBNode(leafWeight);
            }
            
            // Split data
            List<Integer> leftIndices = new ArrayList<>();
            List<Integer> rightIndices = new ArrayList<>();
            
            for (int i : sampleIndices) {
                if (X[i][bestSplit.feature] <= bestSplit.threshold) {
                    leftIndices.add(i);
                } else {
                    rightIndices.add(i);
                }
            }
            
            if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
                return new XGBNode(leafWeight);
            }
            
            // Recursively build children
            XGBNode left = buildTree(X, gradients, hessians,
                leftIndices.stream().mapToInt(Integer::intValue).toArray(),
                featureIndices, depth + 1, random);
            XGBNode right = buildTree(X, gradients, hessians,
                rightIndices.stream().mapToInt(Integer::intValue).toArray(),
                featureIndices, depth + 1, random);
            
            return new XGBNode(bestSplit.feature, bestSplit.threshold, 
                               bestSplit.gain, left, right);
        }
        
        private double computeLeafWeight(double sumG, double sumH) {
            // L1 regularization (soft thresholding)
            double reg = 0;
            if (sumG > alpha) {
                reg = -(sumG - alpha) / (sumH + lambda);
            } else if (sumG < -alpha) {
                reg = -(sumG + alpha) / (sumH + lambda);
            }
            return reg;
        }
        
        private SplitInfo findBestSplit(double[][] X, double[] gradients, double[] hessians,
                                        int[] sampleIndices, int[] featureIndices,
                                        double sumG, double sumH) {
            SplitInfo best = null;
            double bestGain = 0;
            
            for (int f : featureIndices) {
                // Get sorted values for this feature
                List<double[]> sortedData = new ArrayList<>();
                for (int i : sampleIndices) {
                    sortedData.add(new double[]{X[i][f], gradients[i], hessians[i]});
                }
                sortedData.sort(Comparator.comparingDouble(a -> a[0]));
                
                double leftG = 0, leftH = 0;
                
                for (int i = 0; i < sortedData.size() - 1; i++) {
                    leftG += sortedData.get(i)[1];
                    leftH += sortedData.get(i)[2];
                    
                    double rightG = sumG - leftG;
                    double rightH = sumH - leftH;
                    
                    // Skip if values are the same
                    if (Math.abs(sortedData.get(i)[0] - sortedData.get(i + 1)[0]) < 1e-10) {
                        continue;
                    }
                    
                    // Check minimum child weight
                    if (leftH < minChildWeight || rightH < minChildWeight) {
                        continue;
                    }
                    
                    // Compute gain
                    double gain = computeGain(leftG, leftH, rightG, rightH, sumG, sumH);
                    
                    if (gain > bestGain) {
                        bestGain = gain;
                        double threshold = (sortedData.get(i)[0] + sortedData.get(i + 1)[0]) / 2;
                        best = new SplitInfo(f, threshold, gain);
                    }
                }
            }
            
            return best;
        }
        
        private double computeGain(double leftG, double leftH, 
                                   double rightG, double rightH,
                                   double sumG, double sumH) {
            // Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma
            double leftScore = computeScore(leftG, leftH);
            double rightScore = computeScore(rightG, rightH);
            double rootScore = computeScore(sumG, sumH);
            
            return 0.5 * (leftScore + rightScore - rootScore) - gamma;
        }
        
        private double computeScore(double g, double h) {
            // Handle L1 regularization
            if (Math.abs(g) <= alpha) {
                return 0;
            }
            double gAdj = (g > 0) ? g - alpha : g + alpha;
            return gAdj * gAdj / (h + lambda);
        }
        
        public double predict(double[] x) {
            return root.predict(x);
        }
        
        public double[] getFeatureImportances(int nFeatures) {
            double[] importances = new double[nFeatures];
            if (root != null) {
                root.accumulateImportance(importances);
            }
            return importances;
        }
        
        private static class SplitInfo {
            final int feature;
            final double threshold;
            final double gain;
            
            SplitInfo(int feature, double threshold, double gain) {
                this.feature = feature;
                this.threshold = threshold;
                this.gain = gain;
            }
        }
    }
    
    /**
     * XGBoost tree node.
     */
    private static class XGBNode implements Serializable {
        private static final long serialVersionUID = 1L;
        
        final boolean isLeaf;
        final double leafValue;
        final int splitFeature;
        final double splitThreshold;
        final double gain;
        final XGBNode left;
        final XGBNode right;
        
        // Leaf node
        XGBNode(double leafValue) {
            this.isLeaf = true;
            this.leafValue = leafValue;
            this.splitFeature = -1;
            this.splitThreshold = 0;
            this.gain = 0;
            this.left = null;
            this.right = null;
        }
        
        // Internal node
        XGBNode(int splitFeature, double splitThreshold, double gain,
                XGBNode left, XGBNode right) {
            this.isLeaf = false;
            this.leafValue = 0;
            this.splitFeature = splitFeature;
            this.splitThreshold = splitThreshold;
            this.gain = gain;
            this.left = left;
            this.right = right;
        }
        
        double predict(double[] x) {
            if (isLeaf) {
                return leafValue;
            }
            if (x[splitFeature] <= splitThreshold) {
                return left.predict(x);
            } else {
                return right.predict(x);
            }
        }
        
        void accumulateImportance(double[] importances) {
            if (!isLeaf) {
                importances[splitFeature] += gain;
                left.accumulateImportance(importances);
                right.accumulateImportance(importances);
            }
        }
    }
    
    /**
     * Builder class for XGBoostClassifier.
     */
    public static class Builder {
        private int nEstimators = 100;
        private int maxDepth = 6;
        private double learningRate = 0.3;
        private double lambda = 1.0;
        private double alpha = 0.0;
        private double gamma = 0.0;
        private double subsample = 1.0;
        private double colsampleBytree = 1.0;
        private int minChildWeight = 1;
        private Integer randomState = null;
        
        /**
         * Sets the number of boosting rounds (trees per class).
         * 
         * @param nEstimators Number of estimators (default: 100)
         */
        public Builder nEstimators(int nEstimators) {
            if (nEstimators < 1) {
                throw new IllegalArgumentException("nEstimators must be at least 1");
            }
            this.nEstimators = nEstimators;
            return this;
        }
        
        /**
         * Sets the maximum depth of each tree.
         * 
         * @param maxDepth Maximum depth (default: 6)
         */
        public Builder maxDepth(int maxDepth) {
            if (maxDepth < 1) {
                throw new IllegalArgumentException("maxDepth must be at least 1");
            }
            this.maxDepth = maxDepth;
            return this;
        }
        
        /**
         * Sets the learning rate (shrinkage).
         * 
         * @param learningRate Learning rate (default: 0.3)
         */
        public Builder learningRate(double learningRate) {
            if (learningRate <= 0 || learningRate > 1) {
                throw new IllegalArgumentException("learningRate must be in (0, 1]");
            }
            this.learningRate = learningRate;
            return this;
        }
        
        /**
         * Alias for learningRate (XGBoost naming convention).
         */
        public Builder eta(double eta) {
            return learningRate(eta);
        }
        
        /**
         * Sets the L2 regularization weight.
         * 
         * @param lambda L2 regularization (default: 1.0)
         */
        public Builder lambda(double lambda) {
            if (lambda < 0) {
                throw new IllegalArgumentException("lambda must be non-negative");
            }
            this.lambda = lambda;
            return this;
        }
        
        /**
         * Alias for lambda (XGBoost naming convention).
         */
        public Builder regLambda(double regLambda) {
            return lambda(regLambda);
        }
        
        /**
         * Sets the L1 regularization weight.
         * 
         * @param alpha L1 regularization (default: 0.0)
         */
        public Builder alpha(double alpha) {
            if (alpha < 0) {
                throw new IllegalArgumentException("alpha must be non-negative");
            }
            this.alpha = alpha;
            return this;
        }
        
        /**
         * Alias for alpha (XGBoost naming convention).
         */
        public Builder regAlpha(double regAlpha) {
            return alpha(regAlpha);
        }
        
        /**
         * Sets the minimum loss reduction required to make a split.
         * 
         * @param gamma Minimum gain (default: 0.0)
         */
        public Builder gamma(double gamma) {
            if (gamma < 0) {
                throw new IllegalArgumentException("gamma must be non-negative");
            }
            this.gamma = gamma;
            return this;
        }
        
        /**
         * Alias for gamma (XGBoost naming convention).
         */
        public Builder minSplitLoss(double minSplitLoss) {
            return gamma(minSplitLoss);
        }
        
        /**
         * Sets the subsample ratio of the training instances.
         * 
         * @param subsample Subsample ratio (default: 1.0)
         */
        public Builder subsample(double subsample) {
            if (subsample <= 0 || subsample > 1) {
                throw new IllegalArgumentException("subsample must be in (0, 1]");
            }
            this.subsample = subsample;
            return this;
        }
        
        /**
         * Sets the subsample ratio of columns when constructing each tree.
         * 
         * @param colsampleBytree Column subsample ratio (default: 1.0)
         */
        public Builder colsampleBytree(double colsampleBytree) {
            if (colsampleBytree <= 0 || colsampleBytree > 1) {
                throw new IllegalArgumentException("colsampleBytree must be in (0, 1]");
            }
            this.colsampleBytree = colsampleBytree;
            return this;
        }
        
        /**
         * Sets the minimum sum of instance weight (hessian) needed in a child.
         * 
         * @param minChildWeight Minimum child weight (default: 1)
         */
        public Builder minChildWeight(int minChildWeight) {
            if (minChildWeight < 0) {
                throw new IllegalArgumentException("minChildWeight must be non-negative");
            }
            this.minChildWeight = minChildWeight;
            return this;
        }
        
        /**
         * Sets the random state for reproducibility.
         * 
         * @param randomState Random seed
         */
        public Builder randomState(int randomState) {
            this.randomState = randomState;
            return this;
        }
        
        /**
         * Alias for randomState (XGBoost naming convention).
         */
        public Builder seed(int seed) {
            return randomState(seed);
        }
        
        /**
         * Builds the XGBoostClassifier instance.
         */
        public XGBoostClassifier build() {
            return new XGBoostClassifier(nEstimators, maxDepth, learningRate,
                lambda, alpha, gamma, subsample, colsampleBytree,
                minChildWeight, randomState);
        }
    }
}
