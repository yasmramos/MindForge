package io.github.yasmramos.mindforge.regression;

import java.io.Serializable;
import java.util.Random;

/**
 * Stochastic Gradient Descent regressor.
 * Supports different loss functions and regularization.
 */
public class SGDRegressor implements Regressor, Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum Loss { SQUARED, HUBER, EPSILON_INSENSITIVE }
    public enum Penalty { NONE, L1, L2, ELASTICNET }
    
    private final Loss loss;
    private final Penalty penalty;
    private final double alpha;
    private final double l1Ratio;
    private final double epsilon;
    private final double learningRate;
    private final int maxIterations;
    private final double tol;
    private final long randomSeed;
    
    private double[] weights;
    private double bias;
    
    public SGDRegressor(Loss loss, Penalty penalty, double alpha, double l1Ratio,
                       double epsilon, double learningRate, int maxIterations, 
                       double tol, long randomSeed) {
        this.loss = loss;
        this.penalty = penalty;
        this.alpha = alpha;
        this.l1Ratio = l1Ratio;
        this.epsilon = epsilon;
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.tol = tol;
        this.randomSeed = randomSeed;
    }
    
    public SGDRegressor() {
        this(Loss.SQUARED, Penalty.L2, 0.0001, 0.15, 0.1, 0.01, 1000, 1e-4, 42);
    }
    
    @Override
    public void fit(double[][] X, double[] y) {
        Random random = new Random(randomSeed);
        int n = X.length;
        int nFeatures = X[0].length;
        
        weights = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++) {
            weights[j] = random.nextGaussian() * 0.01;
        }
        bias = 0;
        
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        
        double prevLoss = Double.MAX_VALUE;
        
        for (int epoch = 0; epoch < maxIterations; epoch++) {
            shuffle(indices, random);
            double totalLoss = 0;
            
            for (int idx : indices) {
                double[] x = X[idx];
                double yi = y[idx];
                
                double pred = predict(x);
                double error = pred - yi;
                double lossVal = computeLoss(error);
                totalLoss += lossVal;
                
                double dloss = computeLossGradient(error);
                double eta = learningRate / (1 + learningRate * alpha * epoch);
                
                for (int j = 0; j < nFeatures; j++) {
                    double grad = dloss * x[j];
                    grad += computePenaltyGradient(weights[j]);
                    weights[j] -= eta * grad;
                }
                bias -= eta * dloss;
            }
            
            totalLoss /= n;
            if (Math.abs(prevLoss - totalLoss) < tol) break;
            prevLoss = totalLoss;
        }
    }
    
    private double computeLoss(double error) {
        switch (loss) {
            case SQUARED:
                return 0.5 * error * error;
            case HUBER:
                double absError = Math.abs(error);
                if (absError <= epsilon) {
                    return 0.5 * error * error;
                } else {
                    return epsilon * absError - 0.5 * epsilon * epsilon;
                }
            case EPSILON_INSENSITIVE:
                return Math.max(0, Math.abs(error) - epsilon);
            default:
                return 0;
        }
    }
    
    private double computeLossGradient(double error) {
        switch (loss) {
            case SQUARED:
                return error;
            case HUBER:
                if (Math.abs(error) <= epsilon) {
                    return error;
                } else {
                    return epsilon * Math.signum(error);
                }
            case EPSILON_INSENSITIVE:
                if (Math.abs(error) <= epsilon) {
                    return 0;
                } else {
                    return Math.signum(error);
                }
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
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    private double predict(double[] x) {
        double sum = bias;
        for (int j = 0; j < weights.length; j++) {
            sum += weights[j] * x[j];
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
    
    public double[] getWeights() { return weights; }
    public double getBias() { return bias; }
    
    public static class Builder {
        private Loss loss = Loss.SQUARED;
        private Penalty penalty = Penalty.L2;
        private double alpha = 0.0001;
        private double l1Ratio = 0.15;
        private double epsilon = 0.1;
        private double learningRate = 0.01;
        private int maxIterations = 1000;
        private double tol = 1e-4;
        private long randomSeed = 42;
        
        public Builder loss(Loss l) { this.loss = l; return this; }
        public Builder penalty(Penalty p) { this.penalty = p; return this; }
        public Builder alpha(double a) { this.alpha = a; return this; }
        public Builder l1Ratio(double r) { this.l1Ratio = r; return this; }
        public Builder epsilon(double e) { this.epsilon = e; return this; }
        public Builder learningRate(double lr) { this.learningRate = lr; return this; }
        public Builder maxIterations(int n) { this.maxIterations = n; return this; }
        public Builder tol(double t) { this.tol = t; return this; }
        public Builder randomSeed(long s) { this.randomSeed = s; return this; }
        public SGDRegressor build() {
            return new SGDRegressor(loss, penalty, alpha, l1Ratio, epsilon, learningRate, maxIterations, tol, randomSeed);
        }
    }
}
