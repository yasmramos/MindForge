package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.classification.KNearestNeighbors;
import io.github.yasmramos.mindforge.classification.DecisionTreeClassifier;
import io.github.yasmramos.mindforge.classification.GaussianNaiveBayes;
import io.github.yasmramos.mindforge.preprocessing.StandardScaler;
import io.github.yasmramos.mindforge.preprocessing.DataSplit;
import io.github.yasmramos.mindforge.validation.Metrics;

/**
 * Quick Start Guide for MindForge ML Library
 * 
 * This example demonstrates the basic workflow:
 * 1. Prepare data
 * 2. Split into train/test sets
 * 3. Preprocess (scale) the data
 * 4. Train a classifier
 * 5. Make predictions
 * 6. Evaluate performance
 */
public class QuickStart {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge ML Library - Quick Start Guide");
        System.out.println("=".repeat(60));

        // Sample dataset: Iris-like data (4 features, 3 classes)
        double[][] X = {
            // Setosa (class 0)
            {5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
            {4.6, 3.1, 1.5, 0.2}, {5.0, 3.6, 1.4, 0.2}, {5.4, 3.9, 1.7, 0.4},
            {4.6, 3.4, 1.4, 0.3}, {5.0, 3.4, 1.5, 0.2}, {4.4, 2.9, 1.4, 0.2},
            {4.9, 3.1, 1.5, 0.1}, {5.4, 3.7, 1.5, 0.2}, {4.8, 3.4, 1.6, 0.2},
            // Versicolor (class 1)
            {7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
            {5.5, 2.3, 4.0, 1.3}, {6.5, 2.8, 4.6, 1.5}, {5.7, 2.8, 4.5, 1.3},
            {6.3, 3.3, 4.7, 1.6}, {4.9, 2.4, 3.3, 1.0}, {6.6, 2.9, 4.6, 1.3},
            {5.2, 2.7, 3.9, 1.4}, {5.0, 2.0, 3.5, 1.0}, {5.9, 3.0, 4.2, 1.5},
            // Virginica (class 2)
            {6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1},
            {6.3, 2.9, 5.6, 1.8}, {6.5, 3.0, 5.8, 2.2}, {7.6, 3.0, 6.6, 2.1},
            {4.9, 2.5, 4.5, 1.7}, {7.3, 2.9, 6.3, 1.8}, {6.7, 2.5, 5.8, 1.8},
            {7.2, 3.6, 6.1, 2.5}, {6.5, 3.2, 5.1, 2.0}, {6.4, 2.7, 5.3, 1.9}
        };
        
        int[] y = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Setosa
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // Versicolor
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2   // Virginica
        };

        String[] classNames = {"Setosa", "Versicolor", "Virginica"};

        // Step 1: Split data into training and test sets (80/20)
        System.out.println("\n1. Splitting data (80% train, 20% test)...");
        DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.2, true, 42);
        
        double[][] XTrain = split.XTrain;
        double[][] XTest = split.XTest;
        int[] yTrain = split.yTrain;
        int[] yTest = split.yTest;
        
        System.out.printf("   Training samples: %d, Test samples: %d%n", 
                          XTrain.length, XTest.length);

        // Step 2: Scale the data
        System.out.println("\n2. Scaling features with StandardScaler...");
        StandardScaler scaler = new StandardScaler();
        scaler.fit(XTrain);
        XTrain = scaler.transform(XTrain);
        XTest = scaler.transform(XTest);
        System.out.println("   Features scaled to zero mean and unit variance");

        // Step 3: Train classifiers
        System.out.println("\n3. Training classifiers...");
        
        // K-Nearest Neighbors
        KNearestNeighbors knn = new KNearestNeighbors(3);
        knn.train(XTrain, yTrain);
        int[] knnPreds = knn.predict(XTest);
        double knnAcc = Metrics.accuracy(yTest, knnPreds);
        System.out.printf("   KNN (k=3): %.1f%% accuracy%n", knnAcc * 100);

        // Decision Tree
        DecisionTreeClassifier dt = new DecisionTreeClassifier.Builder()
                .maxDepth(5)
                .minSamplesSplit(2)
                .build();
        dt.train(XTrain, yTrain);
        int[] dtPreds = dt.predict(XTest);
        double dtAcc = Metrics.accuracy(yTest, dtPreds);
        System.out.printf("   Decision Tree: %.1f%% accuracy%n", dtAcc * 100);

        // Gaussian Naive Bayes
        GaussianNaiveBayes gnb = new GaussianNaiveBayes();
        gnb.train(XTrain, yTrain);
        int[] gnbPreds = gnb.predict(XTest);
        double gnbAcc = Metrics.accuracy(yTest, gnbPreds);
        System.out.printf("   Gaussian NB: %.1f%% accuracy%n", gnbAcc * 100);

        // Step 4: Find best model and show predictions
        String bestModel = knnAcc >= dtAcc && knnAcc >= gnbAcc ? "KNN" : 
                          (dtAcc >= gnbAcc ? "Decision Tree" : "Gaussian NB");
        int[] bestPreds = knnAcc >= dtAcc && knnAcc >= gnbAcc ? knnPreds : 
                         (dtAcc >= gnbAcc ? dtPreds : gnbPreds);
        double bestAcc = Math.max(knnAcc, Math.max(dtAcc, gnbAcc));

        System.out.println("\n4. Best Model (" + bestModel + ") Predictions:");
        System.out.println("   " + "-".repeat(45));
        for (int i = 0; i < yTest.length; i++) {
            String actual = classNames[yTest[i]];
            String predicted = classNames[bestPreds[i]];
            String status = yTest[i] == bestPreds[i] ? "OK" : "MISS";
            System.out.printf("   Sample %d: Actual=%-12s Predicted=%-12s [%s]%n", 
                              i + 1, actual, predicted, status);
        }

        // Step 5: Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Summary:");
        System.out.println("=".repeat(60));
        System.out.printf("Best model: %s with %.1f%% accuracy%n", bestModel, bestAcc * 100);
        System.out.println("\nMindForge makes ML in Java simple and intuitive!");
    }
}
