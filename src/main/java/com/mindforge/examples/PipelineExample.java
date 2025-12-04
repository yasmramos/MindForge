package com.mindforge.examples;

import com.mindforge.pipeline.*;
import com.mindforge.preprocessing.*;
import com.mindforge.classification.LogisticRegression;
import com.mindforge.regression.LinearRegression;
import com.mindforge.data.Dataset;
import com.mindforge.data.DatasetLoader;
import com.mindforge.validation.Metrics;

/**
 * Demonstrates ML Pipeline functionality in MindForge.
 * 
 * This example shows:
 * - Building preprocessing + model pipelines
 * - Using ColumnTransformer for mixed data types
 * - Grid search for hyperparameter tuning
 * - Complete end-to-end ML workflow
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class PipelineExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Pipeline Example");
        System.out.println("=".repeat(60));
        
        // 1. Simple Pipeline: Scaler + Classifier
        System.out.println("\n1. Simple Pipeline (Scaler + Classifier):");
        System.out.println("-".repeat(40));
        
        // Load Iris dataset
        Dataset iris = DatasetLoader.loadIris();
        double[][] X = iris.getFeatures();
        int[] y = iris.getLabels();
        
        // Split data
        int trainSize = (int) (X.length * 0.8);
        double[][] trainX = new double[trainSize][];
        int[] trainY = new int[trainSize];
        double[][] testX = new double[X.length - trainSize][];
        int[] testY = new int[X.length - trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            trainX[i] = X[i];
            trainY[i] = y[i];
        }
        for (int i = trainSize; i < X.length; i++) {
            testX[i - trainSize] = X[i];
            testY[i - trainSize] = y[i];
        }
        
        // Create pipeline
        Pipeline pipeline = new Pipeline();
        pipeline.addStep("scaler", new StandardScaler());
        pipeline.addStep("classifier", new LogisticRegression(0.1, 1000));
        
        System.out.println("   Pipeline steps:");
        System.out.println("     1. StandardScaler - normalize features");
        System.out.println("     2. LogisticRegression - classification");
        
        // Fit pipeline
        System.out.println("\n   Training pipeline...");
        pipeline.fit(trainX, trainY);
        
        // Predict
        int[] predictions = pipeline.predict(testX);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if (predictions[i] == testY[i]) correct++;
        }
        double accuracy = (double) correct / testY.length;
        
        System.out.println("   Test Accuracy: " + String.format("%.4f", accuracy));
        
        // 2. Pipeline with Multiple Preprocessing Steps
        System.out.println("\n2. Multi-Step Preprocessing Pipeline:");
        System.out.println("-".repeat(40));
        
        Pipeline multiPipeline = new Pipeline();
        multiPipeline.addStep("minmax", new MinMaxScaler());
        multiPipeline.addStep("poly", new PolynomialFeatures(2, false, true));
        multiPipeline.addStep("standard", new StandardScaler());
        multiPipeline.addStep("classifier", new LogisticRegression(0.01, 500));
        
        System.out.println("   Pipeline steps:");
        System.out.println("     1. MinMaxScaler - scale to [0,1]");
        System.out.println("     2. PolynomialFeatures - add interactions");
        System.out.println("     3. StandardScaler - standardize");
        System.out.println("     4. LogisticRegression - classify");
        
        multiPipeline.fit(trainX, trainY);
        int[] multiPredictions = multiPipeline.predict(testX);
        
        correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if (multiPredictions[i] == testY[i]) correct++;
        }
        accuracy = (double) correct / testY.length;
        
        System.out.println("\n   Test Accuracy: " + String.format("%.4f", accuracy));
        
        // 3. ColumnTransformer for Mixed Features
        System.out.println("\n3. ColumnTransformer for Mixed Features:");
        System.out.println("-".repeat(40));
        
        // Simulate mixed data: numeric + categorical-like columns
        System.out.println("   Scenario: Different preprocessing for different columns");
        
        // Columns 0,1 -> StandardScaler, Columns 2,3 -> MinMaxScaler
        int[] numericCols = {0, 1};
        int[] categoricalCols = {2, 3};
        
        ColumnTransformer colTransformer = new ColumnTransformer();
        colTransformer.addTransformer("numeric_scaler", new StandardScaler(), numericCols);
        colTransformer.addTransformer("other_scaler", new MinMaxScaler(), categoricalCols);
        
        System.out.println("   Configuration:");
        System.out.println("     Columns [0,1] -> StandardScaler");
        System.out.println("     Columns [2,3] -> MinMaxScaler");
        
        // Fit and transform
        colTransformer.fit(trainX);
        double[][] transformedTrain = colTransformer.transform(trainX);
        double[][] transformedTest = colTransformer.transform(testX);
        
        System.out.println("\n   Original shape: " + trainX.length + " x " + trainX[0].length);
        System.out.println("   Transformed shape: " + transformedTrain.length + " x " + transformedTrain[0].length);
        
        // Use with classifier
        LogisticRegression clf = new LogisticRegression(0.1, 1000);
        clf.fit(transformedTrain, trainY);
        int[] colPredictions = clf.predict(transformedTest);
        
        correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if (colPredictions[i] == testY[i]) correct++;
        }
        accuracy = (double) correct / testY.length;
        
        System.out.println("   Test Accuracy: " + String.format("%.4f", accuracy));
        
        // 4. GridSearchCV for Hyperparameter Tuning
        System.out.println("\n4. GridSearchCV - Hyperparameter Tuning:");
        System.out.println("-".repeat(40));
        
        System.out.println("   Searching best parameters for LogisticRegression...");
        
        // Define parameter grid
        double[] learningRates = {0.001, 0.01, 0.1};
        int[] iterations = {100, 500, 1000};
        
        System.out.println("   Parameter Grid:");
        System.out.println("     learning_rate: [0.001, 0.01, 0.1]");
        System.out.println("     max_iterations: [100, 500, 1000]");
        System.out.println("     Total combinations: " + (learningRates.length * iterations.length));
        
        // Manual grid search (simulated)
        double bestAccuracy = 0;
        double bestLR = 0;
        int bestIter = 0;
        
        StandardScaler searchScaler = new StandardScaler();
        searchScaler.fit(trainX);
        double[][] scaledTrainX = searchScaler.transform(trainX);
        double[][] scaledTestX = searchScaler.transform(testX);
        
        System.out.println("\n   Searching...");
        for (double lr : learningRates) {
            for (int iter : iterations) {
                LogisticRegression model = new LogisticRegression(lr, iter);
                model.fit(scaledTrainX, trainY);
                int[] preds = model.predict(scaledTestX);
                
                int c = 0;
                for (int i = 0; i < testY.length; i++) {
                    if (preds[i] == testY[i]) c++;
                }
                double acc = (double) c / testY.length;
                
                if (acc > bestAccuracy) {
                    bestAccuracy = acc;
                    bestLR = lr;
                    bestIter = iter;
                }
            }
        }
        
        System.out.println("\n   Best Parameters Found:");
        System.out.println("     learning_rate: " + bestLR);
        System.out.println("     max_iterations: " + bestIter);
        System.out.println("     Best Accuracy: " + String.format("%.4f", bestAccuracy));
        
        // 5. Complete ML Workflow Summary
        System.out.println("\n5. Complete ML Workflow:");
        System.out.println("-".repeat(40));
        System.out.println("   1. Load Data         -> Dataset class");
        System.out.println("   2. Split Data        -> trainTestSplit()");
        System.out.println("   3. Preprocess        -> Scalers, Encoders");
        System.out.println("   4. Feature Engineer  -> PCA, SelectKBest");
        System.out.println("   5. Build Pipeline    -> Pipeline class");
        System.out.println("   6. Tune Parameters   -> GridSearchCV");
        System.out.println("   7. Train Model       -> fit()");
        System.out.println("   8. Evaluate          -> Metrics, ConfusionMatrix");
        System.out.println("   9. Deploy            -> ModelPersistence");
        
        System.out.println("\n   MindForge provides all these components!");
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Example completed successfully!");
        System.out.println("=".repeat(60));
    }
}
