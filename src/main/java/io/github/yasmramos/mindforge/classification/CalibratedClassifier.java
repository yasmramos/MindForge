package io.github.yasmramos.mindforge.classification;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Probability calibration using Platt scaling or isotonic regression.
 * Wraps a classifier to provide well-calibrated probability estimates.
 */
public class CalibratedClassifier implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public enum Method { SIGMOID, ISOTONIC }
    
    private final Classifier baseClassifier;
    private final Method method;
    private final int cv;
    
    private int[] classes;
    // Platt scaling parameters (per class for multi-class)
    private double[] plattA;
    private double[] plattB;
    // Isotonic regression (per class)
    private double[][] isotonicX;
    private double[][] isotonicY;
    
    public CalibratedClassifier(Classifier baseClassifier, Method method, int cv) {
        this.baseClassifier = baseClassifier;
        this.method = method;
        this.cv = cv;
    }
    
    public CalibratedClassifier(Classifier baseClassifier) {
        this(baseClassifier, Method.SIGMOID, 5);
    }
    
    public void fit(double[][] X, int[] y) {
        classes = Arrays.stream(y).distinct().sorted().toArray();
        int nClasses = classes.length;
        
        // First fit the base classifier on all data
        baseClassifier.fit(X, y);
        
        // Get decision function scores using cross-validation
        double[][] scores = new double[X.length][nClasses];
        int[] binaryY = new int[X.length];
        
        int foldSize = X.length / cv;
        for (int fold = 0; fold < cv; fold++) {
            int start = fold * foldSize;
            int end = (fold == cv - 1) ? X.length : start + foldSize;
            
            // Create train/test split
            int trainSize = X.length - (end - start);
            double[][] XTrain = new double[trainSize][];
            int[] yTrain = new int[trainSize];
            
            int trainIdx = 0;
            for (int i = 0; i < X.length; i++) {
                if (i < start || i >= end) {
                    XTrain[trainIdx] = X[i];
                    yTrain[trainIdx++] = y[i];
                }
            }
            
            // Train on this fold
            baseClassifier.fit(XTrain, yTrain);
            
            // Get scores for held-out samples
            for (int i = start; i < end; i++) {
                double[] proba = getDecisionScores(X[i]);
                scores[i] = proba;
            }
        }
        
        // Fit calibration for each class
        if (method == Method.SIGMOID) {
            fitPlattScaling(scores, y);
        } else {
            fitIsotonicRegression(scores, y);
        }
        
        // Refit base classifier on all data
        baseClassifier.fit(X, y);
    }
    
    private double[] getDecisionScores(double[] x) {
        double[] scores = new double[classes.length];
        // Use predict to get basic scores
        int pred = baseClassifier.predict(new double[][]{x})[0];
        for (int c = 0; c < classes.length; c++) {
            scores[c] = (pred == classes[c]) ? 1.0 : 0.0;
        }
        return scores;
    }
    
    private void fitPlattScaling(double[][] scores, int[] y) {
        int nClasses = classes.length;
        plattA = new double[nClasses];
        plattB = new double[nClasses];
        
        for (int c = 0; c < nClasses; c++) {
            // Binary targets for this class
            double[] s = new double[scores.length];
            double[] t = new double[scores.length];
            for (int i = 0; i < scores.length; i++) {
                s[i] = scores[i][c];
                t[i] = (y[i] == classes[c]) ? 1.0 : 0.0;
            }
            
            // Fit sigmoid: P(y=1|s) = 1 / (1 + exp(A*s + B))
            double A = 0, B = 0;
            double lr = 0.1;
            for (int iter = 0; iter < 100; iter++) {
                double gradA = 0, gradB = 0;
                for (int i = 0; i < s.length; i++) {
                    double p = 1.0 / (1.0 + Math.exp(A * s[i] + B));
                    gradA += (p - t[i]) * s[i];
                    gradB += (p - t[i]);
                }
                A -= lr * gradA / s.length;
                B -= lr * gradB / s.length;
            }
            plattA[c] = A;
            plattB[c] = B;
        }
    }
    
    private void fitIsotonicRegression(double[][] scores, int[] y) {
        int nClasses = classes.length;
        isotonicX = new double[nClasses][];
        isotonicY = new double[nClasses][];
        
        for (int c = 0; c < nClasses; c++) {
            double[] s = new double[scores.length];
            double[] t = new double[scores.length];
            for (int i = 0; i < scores.length; i++) {
                s[i] = scores[i][c];
                t[i] = (y[i] == classes[c]) ? 1.0 : 0.0;
            }
            
            // Sort by scores
            int[] indices = new int[s.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            for (int i = 0; i < indices.length - 1; i++) {
                for (int j = i + 1; j < indices.length; j++) {
                    if (s[indices[j]] < s[indices[i]]) {
                        int tmp = indices[i];
                        indices[i] = indices[j];
                        indices[j] = tmp;
                    }
                }
            }
            
            // Pool Adjacent Violators (PAV) algorithm
            double[] sortedS = new double[s.length];
            double[] sortedT = new double[s.length];
            for (int i = 0; i < indices.length; i++) {
                sortedS[i] = s[indices[i]];
                sortedT[i] = t[indices[i]];
            }
            
            double[] isoY = pavAlgorithm(sortedT);
            isotonicX[c] = sortedS;
            isotonicY[c] = isoY;
        }
    }
    
    private double[] pavAlgorithm(double[] y) {
        int n = y.length;
        double[] result = y.clone();
        int[] weight = new int[n];
        Arrays.fill(weight, 1);
        
        int i = 0;
        while (i < n - 1) {
            if (result[i] > result[i + 1]) {
                // Pool
                double sum = result[i] * weight[i] + result[i + 1] * weight[i + 1];
                int w = weight[i] + weight[i + 1];
                result[i] = sum / w;
                weight[i] = w;
                
                // Remove i+1
                for (int j = i + 1; j < n - 1; j++) {
                    result[j] = result[j + 1];
                    weight[j] = weight[j + 1];
                }
                n--;
                
                // Check backwards
                while (i > 0 && result[i - 1] > result[i]) {
                    i--;
                    sum = result[i] * weight[i] + result[i + 1] * weight[i + 1];
                    w = weight[i] + weight[i + 1];
                    result[i] = sum / w;
                    weight[i] = w;
                    for (int j = i + 1; j < n - 1; j++) {
                        result[j] = result[j + 1];
                        weight[j] = weight[j + 1];
                    }
                    n--;
                }
            } else {
                i++;
            }
        }
        
        // Expand back
        double[] expanded = new double[y.length];
        int idx = 0;
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < weight[j]; k++) {
                expanded[idx++] = result[j];
            }
        }
        return expanded;
    }
    
    public int[] predict(double[][] X) {
        double[][] proba = predictProbability(X);
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            int maxIdx = 0;
            for (int c = 1; c < classes.length; c++) {
                if (proba[i][c] > proba[i][maxIdx]) maxIdx = c;
            }
            predictions[i] = classes[maxIdx];
        }
        return predictions;
    }
    
    public double[][] predictProbability(double[][] X) {
        double[][] probas = new double[X.length][classes.length];
        
        for (int i = 0; i < X.length; i++) {
            double[] scores = getDecisionScores(X[i]);
            
            for (int c = 0; c < classes.length; c++) {
                if (method == Method.SIGMOID) {
                    probas[i][c] = 1.0 / (1.0 + Math.exp(plattA[c] * scores[c] + plattB[c]));
                } else {
                    probas[i][c] = interpolate(isotonicX[c], isotonicY[c], scores[c]);
                }
            }
            
            // Normalize
            double sum = 0;
            for (double p : probas[i]) sum += p;
            if (sum > 0) {
                for (int c = 0; c < classes.length; c++) {
                    probas[i][c] /= sum;
                }
            }
        }
        
        return probas;
    }
    
    private double interpolate(double[] x, double[] y, double val) {
        if (val <= x[0]) return y[0];
        if (val >= x[x.length - 1]) return y[y.length - 1];
        
        for (int i = 0; i < x.length - 1; i++) {
            if (val >= x[i] && val < x[i + 1]) {
                double t = (val - x[i]) / (x[i + 1] - x[i]);
                return y[i] + t * (y[i + 1] - y[i]);
            }
        }
        return y[y.length - 1];
    }
    
    public static class Builder {
        private Classifier baseClassifier;
        private Method method = Method.SIGMOID;
        private int cv = 5;
        
        public Builder baseClassifier(Classifier c) { this.baseClassifier = c; return this; }
        public Builder method(Method m) { this.method = m; return this; }
        public Builder cv(int c) { this.cv = c; return this; }
        public CalibratedClassifier build() { return new CalibratedClassifier(baseClassifier, method, cv); }
    }
}
