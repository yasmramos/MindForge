package io.github.yasmramos.mindforge.anomaly;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Isolation Forest algorithm for anomaly detection.
 * Based on the principle that anomalies are easier to isolate than normal points.
 */
public class IsolationForest implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final int nEstimators;
    private final int maxSamples;
    private final double contamination;
    private final Random random;
    private List<IsolationTree> trees;
    private double threshold;
    private int nFeatures;
    
    public IsolationForest(int nEstimators, int maxSamples, double contamination, long randomSeed) {
        this.nEstimators = nEstimators;
        this.maxSamples = maxSamples;
        this.contamination = contamination;
        this.random = new Random(randomSeed);
        this.trees = new ArrayList<>();
    }
    
    public IsolationForest() {
        this(100, 256, 0.1, 42);
    }
    
    public void fit(double[][] X) {
        this.nFeatures = X[0].length;
        int sampleSize = Math.min(maxSamples, X.length);
        int maxDepth = (int) Math.ceil(Math.log(sampleSize) / Math.log(2));
        
        trees.clear();
        for (int i = 0; i < nEstimators; i++) {
            double[][] sample = subsample(X, sampleSize);
            IsolationTree tree = new IsolationTree(maxDepth, random);
            tree.fit(sample);
            trees.add(tree);
        }
        
        // Calculate threshold based on contamination
        double[] scores = decisionFunction(X);
        double[] sortedScores = scores.clone();
        java.util.Arrays.sort(sortedScores);
        int thresholdIdx = (int) (contamination * X.length);
        this.threshold = sortedScores[thresholdIdx];
    }
    
    public int[] predict(double[][] X) {
        double[] scores = decisionFunction(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = scores[i] < threshold ? -1 : 1; // -1 = anomaly, 1 = normal
        }
        return predictions;
    }
    
    public double[] decisionFunction(double[][] X) {
        double[] scores = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double avgPathLength = 0;
            for (IsolationTree tree : trees) {
                avgPathLength += tree.pathLength(X[i]);
            }
            avgPathLength /= trees.size();
            scores[i] = Math.pow(2, -avgPathLength / averagePathLength(maxSamples));
        }
        // Invert so that higher scores = more normal
        for (int i = 0; i < scores.length; i++) {
            scores[i] = -scores[i];
        }
        return scores;
    }
    
    private double averagePathLength(int n) {
        if (n <= 1) return 0;
        return 2.0 * (Math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n);
    }
    
    private double[][] subsample(double[][] X, int sampleSize) {
        double[][] sample = new double[sampleSize][];
        boolean[] selected = new boolean[X.length];
        int count = 0;
        while (count < sampleSize) {
            int idx = random.nextInt(X.length);
            if (!selected[idx]) {
                selected[idx] = true;
                sample[count++] = X[idx];
            }
        }
        return sample;
    }
    
    private static class IsolationTree implements Serializable {
        private static final long serialVersionUID = 1L;
        private Node root;
        private final int maxDepth;
        private final Random random;
        
        IsolationTree(int maxDepth, Random random) {
            this.maxDepth = maxDepth;
            this.random = random;
        }
        
        void fit(double[][] X) {
            root = buildTree(X, 0);
        }
        
        private Node buildTree(double[][] X, int depth) {
            if (depth >= maxDepth || X.length <= 1) {
                return new Node(X.length);
            }
            
            int nFeatures = X[0].length;
            int splitFeature = random.nextInt(nFeatures);
            
            double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
            for (double[] row : X) {
                min = Math.min(min, row[splitFeature]);
                max = Math.max(max, row[splitFeature]);
            }
            
            if (min == max) {
                return new Node(X.length);
            }
            
            double splitValue = min + random.nextDouble() * (max - min);
            
            List<double[]> leftList = new ArrayList<>();
            List<double[]> rightList = new ArrayList<>();
            for (double[] row : X) {
                if (row[splitFeature] < splitValue) {
                    leftList.add(row);
                } else {
                    rightList.add(row);
                }
            }
            
            Node node = new Node(splitFeature, splitValue);
            node.left = buildTree(leftList.toArray(new double[0][]), depth + 1);
            node.right = buildTree(rightList.toArray(new double[0][]), depth + 1);
            return node;
        }
        
        double pathLength(double[] x) {
            return pathLength(x, root, 0);
        }
        
        private double pathLength(double[] x, Node node, int depth) {
            if (node.isLeaf) {
                return depth + averagePathLength(node.size);
            }
            if (x[node.splitFeature] < node.splitValue) {
                return pathLength(x, node.left, depth + 1);
            } else {
                return pathLength(x, node.right, depth + 1);
            }
        }
        
        private double averagePathLength(int n) {
            if (n <= 1) return 0;
            return 2.0 * (Math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n);
        }
    }
    
    private static class Node implements Serializable {
        private static final long serialVersionUID = 1L;
        boolean isLeaf;
        int size;
        int splitFeature;
        double splitValue;
        Node left, right;
        
        Node(int size) {
            this.isLeaf = true;
            this.size = size;
        }
        
        Node(int splitFeature, double splitValue) {
            this.isLeaf = false;
            this.splitFeature = splitFeature;
            this.splitValue = splitValue;
        }
    }
    
    // Builder pattern
    public static class Builder {
        private int nEstimators = 100;
        private int maxSamples = 256;
        private double contamination = 0.1;
        private long randomSeed = 42;
        
        public Builder nEstimators(int n) { this.nEstimators = n; return this; }
        public Builder maxSamples(int n) { this.maxSamples = n; return this; }
        public Builder contamination(double c) { this.contamination = c; return this; }
        public Builder randomSeed(long s) { this.randomSeed = s; return this; }
        public IsolationForest build() { return new IsolationForest(nEstimators, maxSamples, contamination, randomSeed); }
    }
}
