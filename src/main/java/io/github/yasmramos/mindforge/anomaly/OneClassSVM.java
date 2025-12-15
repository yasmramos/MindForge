package io.github.yasmramos.mindforge.anomaly;

import java.io.Serializable;

/**
 * One-Class SVM for anomaly detection using RBF kernel.
 * Learns a decision boundary around normal data.
 */
public class OneClassSVM implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final double nu;
    private final double gamma;
    private final int maxIterations;
    private double[][] supportVectors;
    private double[] alpha;
    private double rho;
    
    public OneClassSVM(double nu, double gamma, int maxIterations) {
        this.nu = nu;
        this.gamma = gamma;
        this.maxIterations = maxIterations;
    }
    
    public OneClassSVM() {
        this(0.1, 0.1, 1000);
    }
    
    public void fit(double[][] X) {
        int n = X.length;
        alpha = new double[n];
        double initialAlpha = 1.0 / n;
        for (int i = 0; i < n; i++) {
            alpha[i] = initialAlpha;
        }
        
        // SMO-style optimization
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double Kii = rbfKernel(X[i], X[i]);
                    double Kjj = rbfKernel(X[j], X[j]);
                    double Kij = rbfKernel(X[i], X[j]);
                    
                    double eta = Kii + Kjj - 2 * Kij;
                    if (eta <= 0) continue;
                    
                    double Ei = decisionValue(X[i], X) - 1;
                    double Ej = decisionValue(X[j], X) - 1;
                    
                    double oldAlphaI = alpha[i];
                    double oldAlphaJ = alpha[j];
                    
                    double newAlphaJ = oldAlphaJ + (Ei - Ej) / eta;
                    double sum = oldAlphaI + oldAlphaJ;
                    
                    newAlphaJ = Math.max(0, Math.min(1.0 / (nu * n), newAlphaJ));
                    double newAlphaI = sum - newAlphaJ;
                    newAlphaI = Math.max(0, Math.min(1.0 / (nu * n), newAlphaI));
                    
                    if (Math.abs(newAlphaI - oldAlphaI) > 1e-5) {
                        alpha[i] = newAlphaI;
                        alpha[j] = newAlphaJ;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
        }
        
        // Extract support vectors
        int svCount = 0;
        for (double a : alpha) {
            if (a > 1e-5) svCount++;
        }
        supportVectors = new double[svCount][];
        double[] svAlpha = new double[svCount];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            if (alpha[i] > 1e-5) {
                supportVectors[idx] = X[i];
                svAlpha[idx] = alpha[i];
                idx++;
            }
        }
        this.alpha = svAlpha;
        
        // Compute rho
        rho = 0;
        for (int i = 0; i < supportVectors.length; i++) {
            rho += decisionValueSV(supportVectors[i]);
        }
        rho /= supportVectors.length;
    }
    
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double score = decisionValueSV(X[i]) - rho;
            predictions[i] = score >= 0 ? 1 : -1;
        }
        return predictions;
    }
    
    public double[] decisionFunction(double[][] X) {
        double[] scores = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            scores[i] = decisionValueSV(X[i]) - rho;
        }
        return scores;
    }
    
    private double decisionValue(double[] x, double[][] X) {
        double sum = 0;
        for (int i = 0; i < X.length; i++) {
            sum += alpha[i] * rbfKernel(x, X[i]);
        }
        return sum;
    }
    
    private double decisionValueSV(double[] x) {
        double sum = 0;
        for (int i = 0; i < supportVectors.length; i++) {
            sum += alpha[i] * rbfKernel(x, supportVectors[i]);
        }
        return sum;
    }
    
    private double rbfKernel(double[] x1, double[] x2) {
        double sum = 0;
        for (int i = 0; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        return Math.exp(-gamma * sum);
    }
    
    public static class Builder {
        private double nu = 0.1;
        private double gamma = 0.1;
        private int maxIterations = 1000;
        
        public Builder nu(double nu) { this.nu = nu; return this; }
        public Builder gamma(double gamma) { this.gamma = gamma; return this; }
        public Builder maxIterations(int n) { this.maxIterations = n; return this; }
        public OneClassSVM build() { return new OneClassSVM(nu, gamma, maxIterations); }
    }
}
