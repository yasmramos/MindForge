package com.mindforge.classification;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Gradient Boosting Classifier implementation.
 * 
 * Builds an ensemble of weak learners (decision stumps) sequentially,
 * where each learner tries to correct the errors of the previous ones.
 * Uses gradient descent to minimize the loss function.
 * 
 * Features:
 * - Configurable number of estimators
 * - Learning rate control
 * - Maximum depth for base learners
 * - Subsampling for stochastic gradient boosting
 * - Multi-class classification via One-vs-Rest
 * 
 * @author MindForge Team
 * @version 1.0.7-alpha
 */
public class GradientBoostingClassifier {
    
    private int nEstimators;
    private double learningRate;
    private int maxDepth;
    private double subsample;
    private Integer randomState;
    private Random random;
    
    // Model storage: one list of trees per class (One-vs-Rest)
    private Map<Integer, List<DecisionStump>> classifierTrees;
    private Map<Integer, double[]> initialPredictions;
    private int[] classes;
    private boolean isTrained;
    
    /**
     * Decision Stump - weak learner for gradient boosting.
     */
    private static class DecisionStump {
        int featureIndex;
        double threshold;
        double leftValue;
        double rightValue;
        
        // For deeper trees
        DecisionStump leftChild;
        DecisionStump rightChild;
        boolean isLeaf;
        double leafValue;
        
        double predict(double[] sample) {
            if (isLeaf) {
                return leafValue;
            }
            if (sample[featureIndex] <= threshold) {
                return leftChild != null ? leftChild.predict(sample) : leftValue;
            } else {
                return rightChild != null ? rightChild.predict(sample) : rightValue;
            }
        }
    }
    
    /**
     * Creates a Gradient Boosting Classifier with default parameters.
     */
    public GradientBoostingClassifier() {
        this(100, 0.1, 3, 1.0, null);
    }
    
    /**
     * Creates a Gradient Boosting Classifier with custom parameters.
     * 
     * @param nEstimators Number of boosting stages (trees)
     * @param learningRate Shrinks the contribution of each tree
     * @param maxDepth Maximum depth of individual trees
     * @param subsample Fraction of samples for fitting trees (stochastic boosting)
     * @param randomState Random seed for reproducibility
     */
    public GradientBoostingClassifier(int nEstimators, double learningRate, 
                                       int maxDepth, double subsample, Integer randomState) {
        if (nEstimators <= 0) {
            throw new IllegalArgumentException("nEstimators must be positive");
        }
        if (learningRate <= 0 || learningRate > 1) {
            throw new IllegalArgumentException("learningRate must be in (0, 1]");
        }
        if (maxDepth <= 0) {
            throw new IllegalArgumentException("maxDepth must be positive");
        }
        if (subsample <= 0 || subsample > 1) {
            throw new IllegalArgumentException("subsample must be in (0, 1]");
        }
        
        this.nEstimators = nEstimators;
        this.learningRate = learningRate;
        this.maxDepth = maxDepth;
        this.subsample = subsample;
        this.randomState = randomState;
        this.random = randomState != null ? new Random(randomState) : new Random();
        this.classifierTrees = new HashMap<>();
        this.initialPredictions = new HashMap<>();
        this.isTrained = false;
    }
    
    /**
     * Trains the Gradient Boosting Classifier.
     * 
     * @param features Training feature matrix
     * @param labels Training labels
     */
    public void fit(double[][] features, int[] labels) {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("Features and labels cannot be null");
        }
        if (features.length != labels.length) {
            throw new IllegalArgumentException("Features and labels must have same length");
        }
        if (features.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        // Extract unique classes
        classes = Arrays.stream(labels).distinct().sorted().toArray();
        
        if (classes.length < 2) {
            throw new IllegalArgumentException("Need at least 2 classes for classification");
        }
        
        int n = features.length;
        int nFeatures = features[0].length;
        
        // Binary or multiclass
        if (classes.length == 2) {
            // Binary classification
            fitBinary(features, labels, classes[1]);
        } else {
            // Multiclass: One-vs-Rest
            for (int cls : classes) {
                fitBinary(features, labels, cls);
            }
        }
        
        isTrained = true;
    }
    
    /**
     * Fits a binary classifier for one class vs rest.
     */
    private void fitBinary(double[][] features, int[] labels, int positiveClass) {
        int n = features.length;
        
        // Convert labels to 0/1 for this class
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = labels[i] == positiveClass ? 1.0 : 0.0;
        }
        
        // Initialize with log-odds
        double posCount = Arrays.stream(y).sum();
        double negCount = n - posCount;
        double initialPred = 0.5; // Start with 0.5 probability
        if (posCount > 0 && negCount > 0) {
            initialPred = posCount / n;
        }
        
        double[] predictions = new double[n];
        Arrays.fill(predictions, logit(initialPred));
        
        initialPredictions.put(positiveClass, new double[]{logit(initialPred)});
        
        List<DecisionStump> trees = new ArrayList<>();
        
        // Boosting iterations
        for (int iter = 0; iter < nEstimators; iter++) {
            // Compute pseudo-residuals (negative gradient of log loss)
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) {
                double prob = sigmoid(predictions[i]);
                residuals[i] = y[i] - prob;
            }
            
            // Subsample if needed
            int[] sampleIndices;
            if (subsample < 1.0) {
                int sampleSize = Math.max(1, (int) (n * subsample));
                sampleIndices = new int[sampleSize];
                for (int i = 0; i < sampleSize; i++) {
                    sampleIndices[i] = random.nextInt(n);
                }
            } else {
                sampleIndices = new int[n];
                for (int i = 0; i < n; i++) {
                    sampleIndices[i] = i;
                }
            }
            
            // Fit a tree to residuals
            DecisionStump tree = fitTree(features, residuals, sampleIndices, 1);
            trees.add(tree);
            
            // Update predictions
            for (int i = 0; i < n; i++) {
                predictions[i] += learningRate * tree.predict(features[i]);
            }
        }
        
        classifierTrees.put(positiveClass, trees);
    }
    
    /**
     * Fits a decision tree (stump or deeper) to the residuals.
     */
    private DecisionStump fitTree(double[][] features, double[] residuals, 
                                   int[] indices, int depth) {
        DecisionStump stump = new DecisionStump();
        
        if (depth >= maxDepth || indices.length <= 1) {
            // Create leaf node
            stump.isLeaf = true;
            stump.leafValue = computeLeafValue(residuals, indices);
            return stump;
        }
        
        int nFeatures = features[0].length;
        double bestGain = Double.NEGATIVE_INFINITY;
        int bestFeature = 0;
        double bestThreshold = 0;
        int[] bestLeftIndices = null;
        int[] bestRightIndices = null;
        
        // Find best split
        for (int f = 0; f < nFeatures; f++) {
            // Get unique thresholds from sample
            double[] values = new double[indices.length];
            for (int i = 0; i < indices.length; i++) {
                values[i] = features[indices[i]][f];
            }
            Arrays.sort(values);
            
            // Try thresholds between sorted values
            for (int t = 0; t < values.length - 1; t++) {
                if (values[t] == values[t + 1]) continue;
                
                double threshold = (values[t] + values[t + 1]) / 2;
                
                // Split indices
                List<Integer> leftList = new ArrayList<>();
                List<Integer> rightList = new ArrayList<>();
                
                for (int idx : indices) {
                    if (features[idx][f] <= threshold) {
                        leftList.add(idx);
                    } else {
                        rightList.add(idx);
                    }
                }
                
                if (leftList.isEmpty() || rightList.isEmpty()) continue;
                
                // Compute gain (variance reduction)
                double gain = computeGain(residuals, indices, 
                    leftList.stream().mapToInt(i -> i).toArray(),
                    rightList.stream().mapToInt(i -> i).toArray());
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = threshold;
                    bestLeftIndices = leftList.stream().mapToInt(i -> i).toArray();
                    bestRightIndices = rightList.stream().mapToInt(i -> i).toArray();
                }
            }
        }
        
        if (bestLeftIndices == null || bestRightIndices == null) {
            // No valid split found, create leaf
            stump.isLeaf = true;
            stump.leafValue = computeLeafValue(residuals, indices);
            return stump;
        }
        
        stump.isLeaf = false;
        stump.featureIndex = bestFeature;
        stump.threshold = bestThreshold;
        
        // Recursively build children
        stump.leftChild = fitTree(features, residuals, bestLeftIndices, depth + 1);
        stump.rightChild = fitTree(features, residuals, bestRightIndices, depth + 1);
        
        return stump;
    }
    
    /**
     * Computes the optimal leaf value for gradient boosting with log loss.
     */
    private double computeLeafValue(double[] residuals, int[] indices) {
        if (indices.length == 0) return 0;
        
        double sum = 0;
        for (int idx : indices) {
            sum += residuals[idx];
        }
        return sum / indices.length;
    }
    
    /**
     * Computes the gain from a split (variance reduction).
     */
    private double computeGain(double[] residuals, int[] parentIndices,
                               int[] leftIndices, int[] rightIndices) {
        double parentVar = computeVariance(residuals, parentIndices);
        double leftVar = computeVariance(residuals, leftIndices);
        double rightVar = computeVariance(residuals, rightIndices);
        
        double leftWeight = (double) leftIndices.length / parentIndices.length;
        double rightWeight = (double) rightIndices.length / parentIndices.length;
        
        return parentVar - (leftWeight * leftVar + rightWeight * rightVar);
    }
    
    /**
     * Computes variance of residuals at given indices.
     */
    private double computeVariance(double[] residuals, int[] indices) {
        if (indices.length == 0) return 0;
        
        double mean = 0;
        for (int idx : indices) {
            mean += residuals[idx];
        }
        mean /= indices.length;
        
        double variance = 0;
        for (int idx : indices) {
            double diff = residuals[idx] - mean;
            variance += diff * diff;
        }
        return variance / indices.length;
    }
    
    /**
     * Predicts class labels for samples.
     * 
     * @param features Feature matrix
     * @return Predicted labels
     */
    public int[] predict(double[][] features) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        int[] predictions = new int[features.length];
        double[][] probas = predictProba(features);
        
        for (int i = 0; i < features.length; i++) {
            int maxIdx = 0;
            double maxProb = probas[i][0];
            for (int j = 1; j < probas[i].length; j++) {
                if (probas[i][j] > maxProb) {
                    maxProb = probas[i][j];
                    maxIdx = j;
                }
            }
            predictions[i] = classes[maxIdx];
        }
        
        return predictions;
    }
    
    /**
     * Predicts a single sample.
     * 
     * @param sample Feature vector
     * @return Predicted label
     */
    public int predict(double[] sample) {
        return predict(new double[][]{sample})[0];
    }
    
    /**
     * Predicts class probabilities for samples.
     * 
     * @param features Feature matrix
     * @return Probability matrix (samples x classes)
     */
    public double[][] predictProba(double[][] features) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        int n = features.length;
        double[][] probas = new double[n][classes.length];
        
        if (classes.length == 2) {
            // Binary classification
            int positiveClass = classes[1];
            List<DecisionStump> trees = classifierTrees.get(positiveClass);
            double initialPred = initialPredictions.get(positiveClass)[0];
            
            for (int i = 0; i < n; i++) {
                double pred = initialPred;
                for (DecisionStump tree : trees) {
                    pred += learningRate * tree.predict(features[i]);
                }
                double prob = sigmoid(pred);
                probas[i][0] = 1 - prob;
                probas[i][1] = prob;
            }
        } else {
            // Multiclass: compute scores for each class
            double[][] scores = new double[n][classes.length];
            
            for (int c = 0; c < classes.length; c++) {
                int cls = classes[c];
                List<DecisionStump> trees = classifierTrees.get(cls);
                double initialPred = initialPredictions.get(cls)[0];
                
                for (int i = 0; i < n; i++) {
                    double pred = initialPred;
                    for (DecisionStump tree : trees) {
                        pred += learningRate * tree.predict(features[i]);
                    }
                    scores[i][c] = pred;
                }
            }
            
            // Softmax to convert to probabilities
            for (int i = 0; i < n; i++) {
                double maxScore = Arrays.stream(scores[i]).max().orElse(0);
                double sumExp = 0;
                for (int c = 0; c < classes.length; c++) {
                    probas[i][c] = Math.exp(scores[i][c] - maxScore);
                    sumExp += probas[i][c];
                }
                for (int c = 0; c < classes.length; c++) {
                    probas[i][c] /= sumExp;
                }
            }
        }
        
        return probas;
    }
    
    /**
     * Predicts probability for a single sample.
     * 
     * @param sample Feature vector
     * @return Probability array for each class
     */
    public double[] predictProba(double[] sample) {
        return predictProba(new double[][]{sample})[0];
    }
    
    /**
     * Sigmoid function.
     */
    private double sigmoid(double x) {
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        } else {
            double expX = Math.exp(x);
            return expX / (1.0 + expX);
        }
    }
    
    /**
     * Logit function (inverse of sigmoid).
     */
    private double logit(double p) {
        p = Math.max(1e-10, Math.min(1 - 1e-10, p));
        return Math.log(p / (1 - p));
    }
    
    // Getters
    
    public int getNEstimators() {
        return nEstimators;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public int getMaxDepth() {
        return maxDepth;
    }
    
    public double getSubsample() {
        return subsample;
    }
    
    public int[] getClasses() {
        return classes != null ? classes.clone() : null;
    }
    
    public int getNumTrees() {
        if (!isTrained) return 0;
        int total = 0;
        for (List<DecisionStump> trees : classifierTrees.values()) {
            total += trees.size();
        }
        return total;
    }
    
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * Builder class for GradientBoostingClassifier.
     */
    public static class Builder {
        private int nEstimators = 100;
        private double learningRate = 0.1;
        private int maxDepth = 3;
        private double subsample = 1.0;
        private Integer randomState = null;
        
        public Builder nEstimators(int nEstimators) {
            this.nEstimators = nEstimators;
            return this;
        }
        
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder maxDepth(int maxDepth) {
            this.maxDepth = maxDepth;
            return this;
        }
        
        public Builder subsample(double subsample) {
            this.subsample = subsample;
            return this;
        }
        
        public Builder randomState(Integer randomState) {
            this.randomState = randomState;
            return this;
        }
        
        public GradientBoostingClassifier build() {
            return new GradientBoostingClassifier(nEstimators, learningRate, 
                                                   maxDepth, subsample, randomState);
        }
    }
}
