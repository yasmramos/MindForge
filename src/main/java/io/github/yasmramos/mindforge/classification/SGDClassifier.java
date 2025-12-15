package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Stochastic Gradient Descent classifier.
 * Supports different loss functions: hinge (SVM), log (logistic regression), perceptron.
 */
public class SGDClassifier implements Classifier, Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum Loss { HINGE, LOG, PERCEPTRON, SQUARED_HINGE }
    public enum Penalty { NONE, L1, L2, ELASTICNET }
    
    private final Loss loss;
    private final Penalty penalty;
    private final double alpha;
    private final double l1Ratio;
    private final double learningRate;
    private final int maxIterations;
    private final double tol;
    private final long randomSeed;
    
    private double[][] weights; // For multi-class
    private double[] biases;
    private int[] classes;
    
    public SGDClassifier(Loss loss, Penalty penalty, double alpha, double l1Ratio,
                        double learningRate, int maxIterations, double tol, long randomSeed) {
        this.loss = loss;
        this.penalty = penalty;
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.tol = tol;
        this.randomSeed = randomSeed;
    }
    
    public SGDClassifier() {
        this(Loss.HINGE, Penalty.L2, 0.0001, 0.15, 0.01, 1000, 1e-4, 42);
    }
    
    @Override
    public void fit(double[][] X, int[] y) {
        Random random = new Random(randomSeed);
        int nFeatures = X[0].length;
        
        // Get unique classes
        classes = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = classes.length;
        
        if (nClasses == 2) {
            // Binary classification
            weights = new double[1][nFeatures];
            biases = new double[1];
            fitBinary(X, y, 0, classes[1], random);
        } else {
            // One-vs-Rest for multi-class
            weights = new double[nClasses][nFeatures];
            biases = new double[nClasses];
            for (int c = 0; c < nClasses; c++) {
                fitBinary(X, y, c, classes[c], random);
            }
        }
    }
    
    private void fitBinary(double[][] X, int[] y, int classIdx, int positiveClass, Random random) {
        int n = X.length;
        int nFeatures = X[0].length;
        double[] w = weights[classIdx];
        
        // Initialize weights
        for (int j = 0; j < nFeatures; j++) {
            w[j] = random.nextGaussian() * 0.01;
        }
        
        int[] binaryY = new int[n];
        for (int i = 0; i < n; i++) {
            binaryY[i] = (y[i] == positiveClass) ? 1 : -1;
        }
        
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        
        double prevLoss = Double.MAX_VALUE;
        
        for (int epoch = 0; epoch < maxIterations; epoch++) {
            shuffle(indices, random);
            double totalLoss = 0;
            
            for (int idx : indices) {
                double[] x = X[idx];
                int yi = binaryY[idx];
                
                double score = dotProduct(w, x) + biases[classIdx];
                double lossVal = computeLoss(score, yi);
                totalLoss += lossVal;
                
                // Compute gradient
                double dloss = computeLossGradient(score, yi);
                
                // Update weights
                double eta = learningRate / (1 + learningRate * alpha * epoch);
                for (int j = 0; j < nFeatures; j++) {
                    double grad = dloss * x[j];
                    grad += computePenaltyGradient(w[j]);
                    w[j] -= eta * grad;
                }
                biases[classIdx] -= eta * dloss;
            }
            
            totalLoss /= n;
            if (Math.abs(prevLoss - totalLoss) < tol) break;
            prevLoss = totalLoss;
        }
    }
    
    private double computeLoss(double score, int y) {
        switch (loss) {
            case HINGE:
                return Math.max(0, 1 - y * score);
            case LOG:
                return Math.log(1 + Math.exp(-y * score));
            case PERCEPTRON:
                return Math.max(0, -y * score);
            case SQUARED_HINGE:
                double h = Math.max(0, 1 - y * score);
                return h * h;
            default:
                return 0;
        }
    }
    
    private double computeLossGradient(double score, int y) {
        switch (loss) {
            case HINGE:
                return (y * score < 1) ? -y : 0;
            case LOG:
                double exp = Math.exp(-y * score);
                return -y * exp / (1 + exp);
            case PERCEPTRON:
                return (y * score < 0) ? -y : 0;
            case SQUARED_HINGE:
                double h = 1 - y * score;
                return (h > 0) ? -2 * y * h : 0;
            default:
                return 0;
        }
    }
    
    private double computePenaltyGradient(double w) {
        switch (penalty) {
            case L2:
                return alpha * w;
            case L1:
                return alpha * Math.signum(w);
            case ELASTICNET:
                return alpha * (l1Ratio * Math.signum(w) + (1 - l1Ratio) * w);
            default:
                return 0;
        }
    }
    
    @Override
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predictOne(X[i]);
        }
        return predictions;
    }
    
    private int predictOne(double[] x) {
        if (classes.length == 2) {
            double score = dotProduct(weights[0], x) + biases[0];
            return score >= 0 ? classes[1] : classes[0];
        } else {
            double maxScore = Double.NEGATIVE_INFINITY;
            int maxClass = classes[0];
            for (int c = 0; c < classes.length; c++) {
                double score = dotProduct(weights[c], x) + biases[c];
                if (score > maxScore) {
                    maxScore = score;
                    maxClass = classes[c];
                }
            }
            return maxClass;
        }
    }
    
    private double dotProduct(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    private void shuffle(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    
    public static class Builder {
        private Loss loss = Loss.HINGE;
        private Penalty penalty = Penalty.L2;
        private double alpha = 0.0001;
        private double l1Ratio = 0.15;
        private double learningRate = 0.01;
        private int maxIterations = 1000;
        private double tol = 1e-4;
        private long randomSeed = 42;
        
        public Builder loss(Loss l) { this.loss = l; return this; }
        public Builder penalty(Penalty p) { this.penalty = p; return this; }
        public Builder alpha(double a) { this.alpha = a; return this; }
        public Builder l1Ratio(double r) { this.l1Ratio = r; return this; }
        public Builder learningRate(double lr) { this.learningRate = lr; return this; }
        public Builder maxIterations(int n) { this.maxIterations = n; return this; }
        public Builder tol(double t) { this.tol = t; return this; }
        public Builder randomSeed(long s) { this.randomSeed = s; return this; }
        public SGDClassifier build() {
            return new SGDClassifier(loss, penalty, alpha, l1Ratio, learningRate, maxIterations, tol, randomSeed);
        }
    }
}
