package io.github.yasmramos.mindforge.multilabel;

import io.github.yasmramos.mindforge.classification.LogisticRegression;

import java.util.*;

/**
 * Multi-Label Classifier using Binary Relevance approach.
 * Trains one binary classifier per label.
 */
public class BinaryRelevanceClassifier {
    
    private int nLabels;
    private int nFeatures;
    private LogisticRegression[] classifiers;
    private double[] labelThresholds;
    
    public BinaryRelevanceClassifier(int nLabels, int nFeatures) {
        this.nLabels = nLabels;
        this.nFeatures = nFeatures;
        this.classifiers = new LogisticRegression[nLabels];
        this.labelThresholds = new double[nLabels];
        Arrays.fill(labelThresholds, 0.5);
        
        for (int i = 0; i < nLabels; i++) {
            classifiers[i] = new LogisticRegression.Builder()
                .maxIter(100)
                .build();
        }
    }
    
    public void fit(double[][] X, int[][] y) {
        for (int label = 0; label < nLabels; label++) {
            int[] yBinary = new int[y.length];
            for (int i = 0; i < y.length; i++) {
                yBinary[i] = y[i][label];
            }
            
            classifiers[label].fit(X, yBinary);
        }
        
        optimizeThresholds(X, y);
    }
    
    private void optimizeThresholds(double[][] X, int[][] y) {
        for (int label = 0; label < nLabels; label++) {
            double bestThreshold = 0.5;
            double bestF1 = 0.0;
            
            for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
                double f1 = computeF1ForLabel(X, y, label, threshold);
                if (f1 > bestF1) {
                    bestF1 = f1;
                    bestThreshold = threshold;
                }
            }
            
            labelThresholds[label] = bestThreshold;
        }
    }
    
    private double computeF1ForLabel(double[][] X, int[][] y, int label, double threshold) {
        int tp = 0, fp = 0, fn = 0;
        
        for (int i = 0; i < X.length; i++) {
            double[][] proba = classifiers[label].predictProba(new double[][]{X[i]});
            double prob = proba[0][1];
            int pred = prob >= threshold ? 1 : 0;
            int actual = y[i][label];
            
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 1) fn++;
        }
        
        double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
        
        return (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
    }
    
    public int[] predict(double[] x) {
        int[] predictions = new int[nLabels];
        for (int label = 0; label < nLabels; label++) {
            double[][] proba = classifiers[label].predictProba(new double[][]{x});
            double prob = proba[0][1];
            predictions[label] = prob >= labelThresholds[label] ? 1 : 0;
        }
        return predictions;
    }
    
    public int[][] predict(double[][] X) {
        int[][] predictions = new int[X.length][nLabels];
        for (int i = 0; i < X.length; i++) {
            predictions[i] = predict(X[i]);
        }
        return predictions;
    }
    
    public double[] predictProba(double[] x) {
        double[] probabilities = new double[nLabels];
        for (int label = 0; label < nLabels; label++) {
            double[][] proba = classifiers[label].predictProba(new double[][]{x});
            probabilities[label] = proba[0][1];
        }
        return probabilities;
    }
    
    public double[][] predictProba(double[][] X) {
        double[][] probabilities = new double[X.length][nLabels];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = predictProba(X[i]);
        }
        return probabilities;
    }
    
    public void setThreshold(int label, double threshold) {
        if (label >= 0 && label < nLabels) {
            labelThresholds[label] = threshold;
        }
    }
}
