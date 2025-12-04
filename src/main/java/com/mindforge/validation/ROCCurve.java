package com.mindforge.validation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * ROC Curve and AUC calculation for binary classification.
 */
public class ROCCurve {
    
    private double[] fpr;  // False Positive Rates
    private double[] tpr;  // True Positive Rates
    private double[] thresholds;
    private double auc;
    
    /**
     * Calculate ROC curve from predictions and true labels.
     * 
     * @param yTrue true binary labels (0 or 1)
     * @param yScores predicted scores/probabilities for the positive class
     */
    public ROCCurve(int[] yTrue, double[] yScores) {
        if (yTrue.length != yScores.length) {
            throw new IllegalArgumentException("Arrays must have the same length");
        }
        
        int n = yTrue.length;
        
        // Create pairs of (score, label) and sort by score descending
        List<double[]> pairs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            pairs.add(new double[]{yScores[i], yTrue[i]});
        }
        pairs.sort((a, b) -> Double.compare(b[0], a[0]));
        
        // Count positives and negatives
        int totalPositives = 0;
        int totalNegatives = 0;
        for (int y : yTrue) {
            if (y == 1) totalPositives++;
            else totalNegatives++;
        }
        
        if (totalPositives == 0 || totalNegatives == 0) {
            throw new IllegalArgumentException("Both classes must be present in y_true");
        }
        
        // Calculate TPR and FPR at each threshold
        List<Double> fprList = new ArrayList<>();
        List<Double> tprList = new ArrayList<>();
        List<Double> thresholdList = new ArrayList<>();
        
        int tp = 0, fp = 0;
        double prevScore = Double.POSITIVE_INFINITY;
        
        fprList.add(0.0);
        tprList.add(0.0);
        thresholdList.add(pairs.get(0)[0] + 1);
        
        for (double[] pair : pairs) {
            double score = pair[0];
            int label = (int) pair[1];
            
            if (score != prevScore) {
                fprList.add((double) fp / totalNegatives);
                tprList.add((double) tp / totalPositives);
                thresholdList.add(score);
                prevScore = score;
            }
            
            if (label == 1) {
                tp++;
            } else {
                fp++;
            }
        }
        
        // Add final point
        fprList.add(1.0);
        tprList.add(1.0);
        thresholdList.add(pairs.get(pairs.size() - 1)[0] - 1);
        
        // Convert to arrays
        fpr = new double[fprList.size()];
        tpr = new double[tprList.size()];
        thresholds = new double[thresholdList.size()];
        
        for (int i = 0; i < fprList.size(); i++) {
            fpr[i] = fprList.get(i);
            tpr[i] = tprList.get(i);
            thresholds[i] = thresholdList.get(i);
        }
        
        // Calculate AUC using trapezoidal rule
        auc = calculateAUC();
    }
    
    /**
     * Calculate AUC using trapezoidal integration.
     * 
     * @return AUC value
     */
    private double calculateAUC() {
        double area = 0.0;
        for (int i = 1; i < fpr.length; i++) {
            double width = fpr[i] - fpr[i - 1];
            double height = (tpr[i] + tpr[i - 1]) / 2.0;
            area += width * height;
        }
        return area;
    }
    
    /**
     * Get the AUC (Area Under Curve).
     * 
     * @return AUC value
     */
    public double getAUC() {
        return auc;
    }
    
    /**
     * Get false positive rates.
     * 
     * @return FPR array
     */
    public double[] getFPR() {
        return fpr;
    }
    
    /**
     * Get true positive rates.
     * 
     * @return TPR array
     */
    public double[] getTPR() {
        return tpr;
    }
    
    /**
     * Get thresholds.
     * 
     * @return threshold array
     */
    public double[] getThresholds() {
        return thresholds;
    }
    
    /**
     * Calculate ROC AUC score.
     * 
     * @param yTrue true binary labels
     * @param yScores predicted scores
     * @return AUC score
     */
    public static double rocAucScore(int[] yTrue, double[] yScores) {
        ROCCurve roc = new ROCCurve(yTrue, yScores);
        return roc.getAUC();
    }
    
    /**
     * Find the optimal threshold using Youden's J statistic.
     * 
     * @return optimal threshold
     */
    public double getOptimalThreshold() {
        double maxJ = Double.NEGATIVE_INFINITY;
        double optimalThreshold = 0.5;
        
        for (int i = 0; i < fpr.length; i++) {
            double j = tpr[i] - fpr[i];  // Youden's J statistic
            if (j > maxJ) {
                maxJ = j;
                optimalThreshold = thresholds[i];
            }
        }
        
        return optimalThreshold;
    }
    
    @Override
    public String toString() {
        return String.format("ROC Curve (AUC = %.4f)", auc);
    }
}
