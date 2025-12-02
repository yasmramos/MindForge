package com.mindforge.examples;

import com.mindforge.validation.CrossValidation;
import com.mindforge.validation.CrossValidationResult;
import com.mindforge.validation.Metrics;
import com.mindforge.classification.KNearestNeighbors;
import com.mindforge.classification.DecisionTreeClassifier;
import com.mindforge.classification.GaussianNaiveBayes;
import java.util.Random;

/**
 * Validation Example with MindForge
 * 
 * Demonstrates model validation techniques:
 * - K-Fold Cross Validation
 * - Evaluation metrics (Accuracy, Precision, Recall, F1)
 */
public class ValidationExample {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Validation Example");
        System.out.println("=".repeat(60));

        // Generate binary classification dataset
        System.out.println("\n1. Generating binary classification dataset...");
        double[][] X = new double[200][4];
        int[] y = new int[200];
        Random rand = new Random(42);
        
        // Class 0: centered around (-1, -1, -1, -1)
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 4; j++) {
                X[i][j] = -1 + rand.nextGaussian() * 0.8;
            }
            y[i] = 0;
        }
        
        // Class 1: centered around (1, 1, 1, 1)
        for (int i = 100; i < 200; i++) {
            for (int j = 0; j < 4; j++) {
                X[i][j] = 1 + rand.nextGaussian() * 0.8;
            }
            y[i] = 1;
        }
        
        System.out.printf("   Created %d samples: %d class-0, %d class-1%n", 
                          X.length, 100, 100);

        // K-Fold Cross Validation
        System.out.println("\n2. K-Fold Cross Validation (k=5):");
        System.out.println("   " + "-".repeat(50));
        System.out.println("   Splits data into 5 folds, trains on 4, tests on 1");
        System.out.println("   Repeats 5 times, each fold serving as test once");
        
        // KNN with K-Fold CV
        CrossValidationResult knnResult = CrossValidation.kFold(
            // Trainer: creates and trains KNN model
            (xTrain, yTrain) -> {
                KNearestNeighbors knn = new KNearestNeighbors(5);
                knn.train(xTrain, yTrain);
                return knn;
            },
            // Predictor: uses model to make predictions
            (model, xTest) -> {
                int[] preds = new int[xTest.length];
                for (int i = 0; i < xTest.length; i++) {
                    preds[i] = model.predict(xTest[i]);
                }
                return preds;
            },
            X, y, 5, 42  // data, labels, k-folds, random seed
        );
        
        System.out.printf("\n   KNN (k=5) Results:%n");
        System.out.printf("     Mean Accuracy: %.2f%% (+/- %.2f%%)%n", 
                          knnResult.getMean() * 100,
                          knnResult.getStdDev() * 100 * 2);
        System.out.printf("     Fold scores: %s%n", formatScores(knnResult.getScores()));

        // Compare models with K-Fold CV
        System.out.println("\n3. Model Comparison using 5-Fold CV:");
        System.out.println("   " + "-".repeat(50));
        
        // Decision Tree
        CrossValidationResult dtResult = CrossValidation.kFold(
            (xTrain, yTrain) -> {
                DecisionTreeClassifier dt = new DecisionTreeClassifier();
                dt.train(xTrain, yTrain);
                return dt;
            },
            (model, xTest) -> {
                int[] preds = new int[xTest.length];
                for (int i = 0; i < xTest.length; i++) {
                    preds[i] = model.predict(xTest[i]);
                }
                return preds;
            },
            X, y, 5, 42
        );
        
        // Gaussian Naive Bayes
        CrossValidationResult nbResult = CrossValidation.kFold(
            (xTrain, yTrain) -> {
                GaussianNaiveBayes nb = new GaussianNaiveBayes();
                nb.train(xTrain, yTrain);
                return nb;
            },
            (model, xTest) -> {
                int[] preds = new int[xTest.length];
                for (int i = 0; i < xTest.length; i++) {
                    preds[i] = model.predict(xTest[i]);
                }
                return preds;
            },
            X, y, 5, 42
        );
        
        System.out.printf("   Model            | Mean Acc  | Std Dev%n");
        System.out.printf("   " + "-".repeat(40) + "%n");
        System.out.printf("   KNN (k=5)        | %.2f%%    | +/- %.2f%%%n", 
                          knnResult.getMean() * 100, knnResult.getStdDev() * 100);
        System.out.printf("   Decision Tree    | %.2f%%    | +/- %.2f%%%n", 
                          dtResult.getMean() * 100, dtResult.getStdDev() * 100);
        System.out.printf("   Gaussian NB      | %.2f%%    | +/- %.2f%%%n", 
                          nbResult.getMean() * 100, nbResult.getStdDev() * 100);

        // Classification Metrics Demo
        System.out.println("\n4. Classification Metrics Demo:");
        System.out.println("   " + "-".repeat(50));
        
        // Simple prediction example
        int[] yTrue = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        int[] yPred = {0, 0, 1, 0, 1, 1, 0, 1, 1, 1};
        
        System.out.println("   True labels:      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]");
        System.out.println("   Predicted labels: [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]");
        
        double accuracy = Metrics.accuracy(yTrue, yPred);
        double precision = Metrics.precision(yTrue, yPred, 1);
        double recall = Metrics.recall(yTrue, yPred, 1);
        double f1 = Metrics.f1Score(yTrue, yPred, 1);
        
        System.out.println("\n   Metrics for class 1 (positive class):");
        System.out.printf("     Accuracy:  %.2f%% (correct predictions / total)%n", accuracy * 100);
        System.out.printf("     Precision: %.2f%% (true pos / predicted pos)%n", precision * 100);
        System.out.printf("     Recall:    %.2f%% (true pos / actual pos)%n", recall * 100);
        System.out.printf("     F1 Score:  %.2f%% (harmonic mean of prec & recall)%n", f1 * 100);

        // Confusion matrix breakdown
        System.out.println("\n   Confusion Matrix Breakdown:");
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1 && yPred[i] == 1) tp++;
            if (yTrue[i] == 0 && yPred[i] == 0) tn++;
            if (yTrue[i] == 0 && yPred[i] == 1) fp++;
            if (yTrue[i] == 1 && yPred[i] == 0) fn++;
        }
        System.out.printf("     True Positives (TP):  %d%n", tp);
        System.out.printf("     True Negatives (TN):  %d%n", tn);
        System.out.printf("     False Positives (FP): %d%n", fp);
        System.out.printf("     False Negatives (FN): %d%n", fn);

        // Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Validation Summary:");
        System.out.println("=".repeat(60));
        System.out.println("- K-Fold CV: More reliable than single train/test split");
        System.out.println("- Standard deviation shows model stability");
        System.out.println("- Accuracy: Overall correctness");
        System.out.println("- Precision: When model says 'positive', how often correct?");
        System.out.println("- Recall: Of all actual positives, how many found?");
        System.out.println("- F1 Score: Balance between precision and recall");
    }
    
    private static String formatScores(double[] scores) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < scores.length; i++) {
            sb.append(String.format("%.2f%%", scores[i] * 100));
            if (i < scores.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
