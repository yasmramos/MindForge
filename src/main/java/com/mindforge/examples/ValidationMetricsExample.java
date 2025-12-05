package com.mindforge.examples;

import com.mindforge.validation.*;
import com.mindforge.data.Dataset;
import com.mindforge.data.DatasetLoader;
import com.mindforge.preprocessing.StandardScaler;

/**
 * Demonstrates validation metrics and model evaluation in MindForge.
 * 
 * This example shows:
 * - Confusion Matrix for classification evaluation
 * - ROC Curve and AUC calculation
 * - Precision, Recall, F1-Score metrics
 * - Cross-validation techniques
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class ValidationMetricsExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Validation Metrics Example");
        System.out.println("=".repeat(60));
        
        // Create sample binary classification data
        System.out.println("\n1. Creating Binary Classification Dataset...");
        int[] yTrue = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0};
        int[] yPred = {0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0};
        
        System.out.println("   Total samples: " + yTrue.length);
        System.out.println("   True labels:      " + arrayToString(yTrue));
        System.out.println("   Predicted labels: " + arrayToString(yPred));
        
        // Confusion Matrix
        System.out.println("\n2. Confusion Matrix Analysis:");
        System.out.println("-".repeat(40));
        ConfusionMatrix cm = new ConfusionMatrix(yTrue, yPred);
        
        int positiveClassIndex = 1;  // Define positive class for binary classification
        System.out.println("   True Positives (TP):  " + cm.getTruePositives(positiveClassIndex));
        System.out.println("   True Negatives (TN):  " + cm.getTrueNegatives(positiveClassIndex));
        System.out.println("   False Positives (FP): " + cm.getFalsePositives(positiveClassIndex));
        System.out.println("   False Negatives (FN): " + cm.getFalseNegatives(positiveClassIndex));
        
        System.out.println("\n   Confusion Matrix:");
        System.out.println("                  Predicted");
        System.out.println("                  0      1");
        System.out.println("   Actual  0    [" + cm.getTrueNegatives(positiveClassIndex) + "]    [" + cm.getFalsePositives(positiveClassIndex) + "]");
        System.out.println("           1    [" + cm.getFalseNegatives(positiveClassIndex) + "]    [" + cm.getTruePositives(positiveClassIndex) + "]");
        
        // Metrics from Confusion Matrix
        System.out.println("\n3. Classification Metrics:");
        System.out.println("-".repeat(40));
        System.out.println("   Accuracy:  " + String.format("%.4f", cm.getAccuracy()));
        System.out.println("   Precision: " + String.format("%.4f", cm.getPrecision(positiveClassIndex)));
        System.out.println("   Recall:    " + String.format("%.4f", cm.getRecall(positiveClassIndex)));
        System.out.println("   F1-Score:  " + String.format("%.4f", cm.getF1Score(positiveClassIndex)));
        
        // Using Metrics class directly with positiveClass parameter
        System.out.println("\n4. Using Metrics Class:");
        System.out.println("-".repeat(40));
        int positiveClass = 1;  // Define positive class for binary classification
        System.out.println("   Accuracy:  " + String.format("%.4f", Metrics.accuracy(yTrue, yPred)));
        System.out.println("   Precision: " + String.format("%.4f", Metrics.precision(yTrue, yPred, positiveClass)));
        System.out.println("   Recall:    " + String.format("%.4f", Metrics.recall(yTrue, yPred, positiveClass)));
        System.out.println("   F1-Score:  " + String.format("%.4f", Metrics.f1Score(yTrue, yPred, positiveClass)));
        
        // ROC Curve and AUC
        System.out.println("\n5. ROC Curve and AUC:");
        System.out.println("-".repeat(40));
        
        // Create probability scores for ROC curve
        double[] yScores = {0.1, 0.2, 0.6, 0.3, 0.15, 0.9, 0.85, 0.4, 0.75, 0.95,
                           0.25, 0.55, 0.8, 0.7, 0.2, 0.35, 0.1, 0.88, 0.92, 0.18};
        
        System.out.println("   Probability scores: " + arrayToString(yScores));
        
        ROCCurve roc = new ROCCurve(yTrue, yScores);
        
        System.out.println("\n   AUC Score: " + String.format("%.4f", roc.getAUC()));
        System.out.println("   Optimal Threshold: " + String.format("%.4f", roc.getOptimalThreshold()));
        
        double[] fpr = roc.getFPR();
        double[] tpr = roc.getTPR();
        double[] thresholds = roc.getThresholds();
        
        System.out.println("\n   ROC Curve Points (sample):");
        System.out.println("   Threshold    FPR      TPR");
        for (int i = 0; i < Math.min(5, thresholds.length); i++) {
            System.out.println("   " + String.format("%.4f", thresholds[i]) + 
                             "     " + String.format("%.4f", fpr[i]) + 
                             "   " + String.format("%.4f", tpr[i]));
        }
        
        // AUC Interpretation
        System.out.println("\n6. AUC Interpretation:");
        System.out.println("-".repeat(40));
        double auc = roc.getAUC();
        String interpretation;
        if (auc >= 0.9) interpretation = "Excellent";
        else if (auc >= 0.8) interpretation = "Good";
        else if (auc >= 0.7) interpretation = "Fair";
        else if (auc >= 0.6) interpretation = "Poor";
        else interpretation = "Failed";
        
        System.out.println("   AUC = " + String.format("%.4f", auc) + " (" + interpretation + ")");
        System.out.println("   - AUC 0.9-1.0: Excellent discrimination");
        System.out.println("   - AUC 0.8-0.9: Good discrimination");
        System.out.println("   - AUC 0.7-0.8: Fair discrimination");
        System.out.println("   - AUC 0.6-0.7: Poor discrimination");
        System.out.println("   - AUC 0.5-0.6: Fail (no discrimination)");
        
        // Multiclass metrics
        System.out.println("\n7. Multiclass Classification:");
        System.out.println("-".repeat(40));
        
        int[] multiTrue = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        int[] multiPred = {0, 1, 1, 0, 2, 2, 0, 1, 0};
        
        ConfusionMatrix multiCM = new ConfusionMatrix(multiTrue, multiPred);
        System.out.println("   3-class confusion matrix created");
        System.out.println("   Overall Accuracy: " + String.format("%.4f", multiCM.getAccuracy()));
        
        // Regression Metrics Demo
        System.out.println("\n8. Regression Metrics:");
        System.out.println("-".repeat(40));
        
        double[] yTrueReg = {3.0, 5.0, 2.5, 7.0, 4.5};
        double[] yPredReg = {2.8, 5.2, 2.3, 6.8, 4.8};
        
        System.out.println("   True values:      " + arrayToString(yTrueReg));
        System.out.println("   Predicted values: " + arrayToString(yPredReg));
        
        System.out.println("\n   MSE:  " + String.format("%.4f", Metrics.mse(yTrueReg, yPredReg)));
        System.out.println("   RMSE: " + String.format("%.4f", Metrics.rmse(yTrueReg, yPredReg)));
        System.out.println("   MAE:  " + String.format("%.4f", Metrics.mae(yTrueReg, yPredReg)));
        System.out.println("   R2:   " + String.format("%.4f", Metrics.r2Score(yTrueReg, yPredReg)));
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Example completed successfully!");
        System.out.println("=".repeat(60));
    }
    
    private static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(arr.length, 10); i++) {
            sb.append(arr[i]);
            if (i < arr.length - 1 && i < 9) sb.append(", ");
        }
        if (arr.length > 10) sb.append("...");
        sb.append("]");
        return sb.toString();
    }
    
    private static String arrayToString(double[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(arr.length, 6); i++) {
            sb.append(String.format("%.2f", arr[i]));
            if (i < arr.length - 1 && i < 5) sb.append(", ");
        }
        if (arr.length > 6) sb.append("...");
        sb.append("]");
        return sb.toString();
    }
}
