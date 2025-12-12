package com.mindforge.model_selection;

import java.util.*;

/**
 * Classification and regression metrics.
 */
public class Metrics {
    
    /**
     * Accuracy score.
     */
    public static double accuracy(int[] yTrue, int[] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }
        int correct = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == yPred[i]) correct++;
        }
        return (double) correct / yTrue.length;
    }
    
    /**
     * Precision score (binary or weighted average for multiclass).
     */
    public static double precision(int[] yTrue, int[] yPred) {
        int[][] cm = confusionMatrix(yTrue, yPred);
        int nClasses = cm.length;
        
        double totalPrecision = 0;
        int totalSamples = 0;
        
        for (int c = 0; c < nClasses; c++) {
            int tp = cm[c][c];
            int fp = 0;
            for (int i = 0; i < nClasses; i++) {
                if (i != c) fp += cm[i][c];
            }
            int support = 0;
            for (int j = 0; j < nClasses; j++) support += cm[c][j];
            
            if (tp + fp > 0) {
                totalPrecision += ((double) tp / (tp + fp)) * support;
            }
            totalSamples += support;
        }
        
        return totalSamples > 0 ? totalPrecision / totalSamples : 0;
    }
    
    /**
     * Recall score (weighted average for multiclass).
     */
    public static double recall(int[] yTrue, int[] yPred) {
        int[][] cm = confusionMatrix(yTrue, yPred);
        int nClasses = cm.length;
        
        double totalRecall = 0;
        int totalSamples = 0;
        
        for (int c = 0; c < nClasses; c++) {
            int tp = cm[c][c];
            int fn = 0;
            for (int j = 0; j < nClasses; j++) {
                if (j != c) fn += cm[c][j];
            }
            int support = tp + fn;
            
            if (support > 0) {
                totalRecall += ((double) tp / support) * support;
            }
            totalSamples += support;
        }
        
        return totalSamples > 0 ? totalRecall / totalSamples : 0;
    }
    
    /**
     * F1 score (weighted average for multiclass).
     */
    public static double f1Score(int[] yTrue, int[] yPred) {
        double p = precision(yTrue, yPred);
        double r = recall(yTrue, yPred);
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0;
    }
    
    /**
     * Confusion matrix.
     */
    public static int[][] confusionMatrix(int[] yTrue, int[] yPred) {
        int maxLabel = 0;
        for (int v : yTrue) maxLabel = Math.max(maxLabel, v);
        for (int v : yPred) maxLabel = Math.max(maxLabel, v);
        
        int nClasses = maxLabel + 1;
        int[][] cm = new int[nClasses][nClasses];
        
        for (int i = 0; i < yTrue.length; i++) {
            cm[yTrue[i]][yPred[i]]++;
        }
        
        return cm;
    }
    
    /**
     * ROC AUC score (binary classification).
     */
    public static double rocAucScore(int[] yTrue, double[] yScores) {
        int n = yTrue.length;
        
        // Count positives and negatives
        int nPos = 0, nNeg = 0;
        for (int y : yTrue) {
            if (y == 1) nPos++;
            else nNeg++;
        }
        
        if (nPos == 0 || nNeg == 0) return 0.5;
        
        // Sort by scores descending
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(yScores[b], yScores[a]));
        
        // Calculate AUC using trapezoidal rule
        double auc = 0;
        double tpCount = 0;
        double fpCount = 0;
        double prevTpr = 0;
        double prevFpr = 0;
        
        for (int i = 0; i < n; i++) {
            if (yTrue[indices[i]] == 1) {
                tpCount++;
            } else {
                fpCount++;
            }
            
            double tpr = tpCount / nPos;
            double fpr = fpCount / nNeg;
            
            auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
            prevTpr = tpr;
            prevFpr = fpr;
        }
        
        return auc;
    }
    
    /**
     * Mean Squared Error.
     */
    public static double meanSquaredError(double[] yTrue, double[] yPred) {
        double sum = 0;
        for (int i = 0; i < yTrue.length; i++) {
            double diff = yTrue[i] - yPred[i];
            sum += diff * diff;
        }
        return sum / yTrue.length;
    }
    
    /**
     * R² score (coefficient of determination).
     */
    public static double r2Score(double[] yTrue, double[] yPred) {
        double mean = 0;
        for (double v : yTrue) mean += v;
        mean /= yTrue.length;
        
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < yTrue.length; i++) {
            ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
            ssTot += (yTrue[i] - mean) * (yTrue[i] - mean);
        }
        
        return ssTot > 0 ? 1 - ssRes / ssTot : 0;
    }
}
