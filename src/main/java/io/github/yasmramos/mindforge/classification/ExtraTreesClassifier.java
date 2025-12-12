package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.*;

/**
 * Extremely Randomized Trees (Extra-Trees) Classifier.
 * 
 * Similar to Random Forest but with additional randomization:
 * - Uses all samples (no bootstrapping by default)
 * - Random split thresholds instead of best splits
 * 
 * This makes Extra-Trees faster and often more resistant to overfitting.
 * 
 * @author MindForge
 */
public class ExtraTreesClassifier implements Classifier<double[]>, ProbabilisticClassifier<double[]>, Serializable {
    private static final long serialVersionUID = 1L;
    
    private int nEstimators;
    private int maxDepth;
    private int minSamplesSplit;
    private int minSamplesLeaf;
    private int maxFeatures;
    private boolean bootstrap;
    private Integer randomState;
    
    private List<ExtraTree> trees;
    private boolean trained;
    private int[] classes;
    private int numClasses;
    private int nFeatures;
    private Random random;
    
    /**
     * Creates an ExtraTreesClassifier with default parameters.
     */
    public ExtraTreesClassifier() {
        this(100, -1, 2, 1, -1, false, null);
    }
    
    /**
     * Creates an ExtraTreesClassifier with specified number of estimators.
     */
    public ExtraTreesClassifier(int nEstimators) {
        this(nEstimators, -1, 2, 1, -1, false, null);
    }
    
    /**
     * Creates an ExtraTreesClassifier with full configuration.
     */
    public ExtraTreesClassifier(int nEstimators, int maxDepth, int minSamplesSplit,
                                 int minSamplesLeaf, int maxFeatures, boolean bootstrap,
                                 Integer randomState) {
        if (nEstimators <= 0) {
            throw new IllegalArgumentException("nEstimators must be positive");
        }
        if (minSamplesSplit < 2) {
            throw new IllegalArgumentException("minSamplesSplit must be >= 2");
        }
        if (minSamplesLeaf < 1) {
            throw new IllegalArgumentException("minSamplesLeaf must be >= 1");
        }
        
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.randomState = randomState;
        this.trained = false;
    }
    
    /**
     * Builder pattern for ExtraTreesClassifier.
     */
    public static class Builder {
        private int nEstimators = 100;
        private int maxDepth = -1;
        private int minSamplesSplit = 2;
        private int minSamplesLeaf = 1;
        private int maxFeatures = -1;
        private boolean bootstrap = false;
        private Integer randomState = null;
        
        public Builder nEstimators(int n) { this.nEstimators = n; return this; }
        public Builder maxDepth(int d) { this.maxDepth = d; return this; }
        public Builder minSamplesSplit(int n) { this.minSamplesSplit = n; return this; }
        public Builder minSamplesLeaf(int n) { this.minSamplesLeaf = n; return this; }
        public Builder maxFeatures(int n) { this.maxFeatures = n; return this; }
        public Builder bootstrap(boolean b) { this.bootstrap = b; return this; }
        public Builder randomState(Integer seed) { this.randomState = seed; return this; }
        
        public ExtraTreesClassifier build() {
            return new ExtraTreesClassifier(nEstimators, maxDepth, minSamplesSplit,
                minSamplesLeaf, maxFeatures, bootstrap, randomState);
        }
    }
    
    @Override
    public void train(double[][] X, int[] y) {
        if (X == null || y == null) {
            throw new IllegalArgumentException("X and y cannot be null");
        }
        if (X.length != y.length) {
            throw new IllegalArgumentException("X and y must have same length");
        }
        if (X.length == 0) {
            throw new IllegalArgumentException("X cannot be empty");
        }
        
        int n = X.length;
        nFeatures = X[0].length;
        
        random = randomState != null ? new Random(randomState) : new Random();
        
        // Determine max features
        int actualMaxFeatures = maxFeatures > 0 ? maxFeatures : 
            (int) Math.sqrt(nFeatures);
        actualMaxFeatures = Math.min(actualMaxFeatures, nFeatures);
        
        // Find unique classes
        Set<Integer> classSet = new TreeSet<>();
        for (int label : y) {
            classSet.add(label);
        }
        numClasses = classSet.size();
        classes = new int[numClasses];
        int idx = 0;
        for (int c : classSet) {
            classes[idx++] = c;
        }
        
        // Train trees
        trees = new ArrayList<>();
        
        for (int t = 0; t < nEstimators; t++) {
            double[][] XSample;
            int[] ySample;
            
            if (bootstrap) {
                // Bootstrap sampling
                int[] indices = new int[n];
                for (int i = 0; i < n; i++) {
                    indices[i] = random.nextInt(n);
                }
                XSample = new double[n][];
                ySample = new int[n];
                for (int i = 0; i < n; i++) {
                    XSample[i] = X[indices[i]];
                    ySample[i] = y[indices[i]];
                }
            } else {
                // Use all samples
                XSample = X;
                ySample = y;
            }
            
            ExtraTree tree = new ExtraTree(maxDepth, minSamplesSplit, minSamplesLeaf,
                actualMaxFeatures, classes, random);
            tree.train(XSample, ySample);
            trees.add(tree);
        }
        
        trained = true;
    }
    
    @Override
    public int predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        if (x == null || x.length != nFeatures) {
            throw new IllegalArgumentException("Invalid input dimensions");
        }
        
        double[] proba = predictProba(x);
        int maxIdx = 0;
        for (int i = 1; i < proba.length; i++) {
            if (proba[i] > proba[maxIdx]) {
                maxIdx = i;
            }
        }
        return classes[maxIdx];
    }
    
    @Override
    public int[] predict(double[][] X) {
        if (X == null) {
            throw new IllegalArgumentException("X cannot be null");
        }
        
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    @Override
    public int getNumClasses() {
        return numClasses;
    }
    
    @Override
    public double[] predictProba(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        double[] proba = new double[numClasses];
        
        for (ExtraTree tree : trees) {
            double[] treeProba = tree.predictProba(x);
            for (int i = 0; i < numClasses; i++) {
                proba[i] += treeProba[i];
            }
        }
        
        for (int i = 0; i < numClasses; i++) {
            proba[i] /= nEstimators;
        }
        
        return proba;
    }
    
    public double score(double[][] X, int[] y) {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    public boolean isTrained() {
        return trained;
    }
    
    /**
     * Computes feature importances based on impurity decrease.
     */
    public double[] getFeatureImportances() {
        if (!trained) {
            throw new IllegalStateException("Model not trained");
        }
        
        double[] importances = new double[nFeatures];
        
        for (ExtraTree tree : trees) {
            double[] treeImportances = tree.getFeatureImportances(nFeatures);
            for (int i = 0; i < nFeatures; i++) {
                importances[i] += treeImportances[i];
            }
        }
        
        // Normalize
        double sum = 0;
        for (double imp : importances) sum += imp;
        if (sum > 0) {
            for (int i = 0; i < nFeatures; i++) {
                importances[i] /= sum;
            }
        }
        
        return importances;
    }
    
    // Getters
    public int getNEstimators() { return nEstimators; }
    public int getMaxDepth() { return maxDepth; }
    public int[] getClasses() { return classes != null ? classes.clone() : null; }
    
    /**
     * Internal Extra Tree implementation.
     */
    private static class ExtraTree implements Serializable {
        private static final long serialVersionUID = 1L;
        
        private Node root;
        private int maxDepth;
        private int minSamplesSplit;
        private int minSamplesLeaf;
        private int maxFeatures;
        private int[] classes;
        private Random random;
        
        ExtraTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                  int maxFeatures, int[] classes, Random random) {
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.minSamplesLeaf = minSamplesLeaf;
            this.maxFeatures = maxFeatures;
            this.classes = classes;
            this.random = random;
        }
        
        void train(double[][] X, int[] y) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < X.length; i++) {
                indices.add(i);
            }
            root = buildTree(X, y, indices, 0);
        }
        
        private Node buildTree(double[][] X, int[] y, List<Integer> indices, int depth) {
            Node node = new Node();
            node.nSamples = indices.size();
            
            // Calculate class distribution
            int[] classCounts = new int[classes.length];
            for (int idx : indices) {
                for (int c = 0; c < classes.length; c++) {
                    if (y[idx] == classes[c]) {
                        classCounts[c]++;
                        break;
                    }
                }
            }
            node.classCounts = classCounts;
            
            // Check stopping conditions
            boolean isPure = false;
            for (int count : classCounts) {
                if (count == indices.size()) {
                    isPure = true;
                    break;
                }
            }
            
            if (isPure || indices.size() < minSamplesSplit || 
                (maxDepth > 0 && depth >= maxDepth)) {
                node.isLeaf = true;
                return node;
            }
            
            // Select random features
            int nFeatures = X[0].length;
            List<Integer> featureIndices = new ArrayList<>();
            for (int i = 0; i < nFeatures; i++) {
                featureIndices.add(i);
            }
            Collections.shuffle(featureIndices, random);
            
            // Find best random split
            double bestScore = Double.NEGATIVE_INFINITY;
            int bestFeature = -1;
            double bestThreshold = 0;
            List<Integer> bestLeft = null, bestRight = null;
            
            for (int f = 0; f < Math.min(maxFeatures, nFeatures); f++) {
                int feature = featureIndices.get(f);
                
                // Find min and max for this feature
                double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
                for (int idx : indices) {
                    min = Math.min(min, X[idx][feature]);
                    max = Math.max(max, X[idx][feature]);
                }
                
                if (min >= max) continue;
                
                // Random threshold
                double threshold = min + random.nextDouble() * (max - min);
                
                // Split
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                for (int idx : indices) {
                    if (X[idx][feature] <= threshold) {
                        left.add(idx);
                    } else {
                        right.add(idx);
                    }
                }
                
                if (left.size() < minSamplesLeaf || right.size() < minSamplesLeaf) {
                    continue;
                }
                
                // Calculate score (negative gini impurity decrease)
                double score = calculateSplitScore(y, indices, left, right);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestFeature = feature;
                    bestThreshold = threshold;
                    bestLeft = left;
                    bestRight = right;
                }
            }
            
            if (bestFeature < 0) {
                node.isLeaf = true;
                return node;
            }
            
            node.featureIndex = bestFeature;
            node.threshold = bestThreshold;
            node.impurityDecrease = bestScore;
            node.left = buildTree(X, y, bestLeft, depth + 1);
            node.right = buildTree(X, y, bestRight, depth + 1);
            
            return node;
        }
        
        private double calculateSplitScore(int[] y, List<Integer> parent,
                                           List<Integer> left, List<Integer> right) {
            double parentGini = calculateGini(y, parent);
            double leftGini = calculateGini(y, left);
            double rightGini = calculateGini(y, right);
            
            double n = parent.size();
            return parentGini - (left.size() / n * leftGini + right.size() / n * rightGini);
        }
        
        private double calculateGini(int[] y, List<Integer> indices) {
            int[] counts = new int[classes.length];
            for (int idx : indices) {
                for (int c = 0; c < classes.length; c++) {
                    if (y[idx] == classes[c]) {
                        counts[c]++;
                        break;
                    }
                }
            }
            
            double gini = 1.0;
            double n = indices.size();
            for (int count : counts) {
                double p = count / n;
                gini -= p * p;
            }
            return gini;
        }
        
        double[] predictProba(double[] x) {
            Node node = root;
            while (!node.isLeaf) {
                if (x[node.featureIndex] <= node.threshold) {
                    node = node.left;
                } else {
                    node = node.right;
                }
            }
            
            double[] proba = new double[classes.length];
            double total = 0;
            for (int count : node.classCounts) total += count;
            
            for (int i = 0; i < classes.length; i++) {
                proba[i] = node.classCounts[i] / total;
            }
            
            return proba;
        }
        
        double[] getFeatureImportances(int nFeatures) {
            double[] importances = new double[nFeatures];
            accumulateImportances(root, importances);
            return importances;
        }
        
        private void accumulateImportances(Node node, double[] importances) {
            if (node == null || node.isLeaf) return;
            
            importances[node.featureIndex] += node.impurityDecrease * node.nSamples;
            accumulateImportances(node.left, importances);
            accumulateImportances(node.right, importances);
        }
        
        private static class Node implements Serializable {
            private static final long serialVersionUID = 1L;
            boolean isLeaf = false;
            int featureIndex;
            double threshold;
            double impurityDecrease;
            int nSamples;
            int[] classCounts;
            Node left, right;
        }
    }
}
