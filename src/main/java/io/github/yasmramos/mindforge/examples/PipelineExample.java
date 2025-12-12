package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.preprocessing.*;
import io.github.yasmramos.mindforge.classification.KNearestNeighbors;
import io.github.yasmramos.mindforge.classification.DecisionTreeClassifier;
import io.github.yasmramos.mindforge.data.Dataset;
import io.github.yasmramos.mindforge.data.DatasetLoader;

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
        
        // Manual pipeline: StandardScaler -> KNN
        System.out.println("   Pipeline steps:");
        System.out.println("     1. StandardScaler - normalize features");
        System.out.println("     2. KNearestNeighbors - classification");
        
        StandardScaler scaler = new StandardScaler();
        scaler.fit(trainX);
        double[][] scaledTrainX = scaler.transform(trainX);
        double[][] scaledTestX = scaler.transform(testX);
        
        KNearestNeighbors knn = new KNearestNeighbors(5);
        knn.train(scaledTrainX, trainY);
        
        // Predict
        int[] predictions = knn.predict(scaledTestX);
        
        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if (predictions[i] == testY[i]) correct++;
        }
        double accuracy = (double) correct / testY.length;
        
        System.out.println("\n   Training pipeline...");
        System.out.println("   Test Accuracy: " + String.format("%.4f", accuracy));
        
        // 2. Pipeline with Multiple Preprocessing Steps
        System.out.println("\n2. Multi-Step Preprocessing Pipeline:");
        System.out.println("-".repeat(40));
        
        // MinMaxScaler -> PolynomialFeatures -> StandardScaler -> KNN
        MinMaxScaler minmax = new MinMaxScaler();
        minmax.fit(trainX);
        double[][] mmScaled = minmax.transform(trainX);
        
        PolynomialFeatures poly = new PolynomialFeatures(2, false, true);
        poly.fit(mmScaled);
        double[][] polyFeatures = poly.transform(mmScaled);
        
        StandardScaler stdScaler = new StandardScaler();
        stdScaler.fit(polyFeatures);
        double[][] finalTrainX = stdScaler.transform(polyFeatures);
        
        // Transform test data through same pipeline
        double[][] mmTestScaled = minmax.transform(testX);
        double[][] polyTestFeatures = poly.transform(mmTestScaled);
        double[][] finalTestX = stdScaler.transform(polyTestFeatures);
        
        System.out.println("   Pipeline steps:");
        System.out.println("     1. MinMaxScaler - scale to [0,1]");
        System.out.println("     2. PolynomialFeatures - add interactions");
        System.out.println("     3. StandardScaler - standardize");
        System.out.println("     4. KNearestNeighbors - classify");
        
        KNearestNeighbors multiKnn = new KNearestNeighbors(5);
        multiKnn.train(finalTrainX, trainY);
        int[] multiPredictions = multiKnn.predict(finalTestX);
        
        correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if (multiPredictions[i] == testY[i]) correct++;
        }
        accuracy = (double) correct / testY.length;
        
        System.out.println("\n   Test Accuracy: " + String.format("%.4f", accuracy));
        
        // 3. Different Scalers for Different Columns
        System.out.println("\n3. Column-wise Preprocessing:");
        System.out.println("-".repeat(40));
        
        // Simulate mixed data: numeric + categorical-like columns
        System.out.println("   Scenario: Different preprocessing for different columns");
        
        // Columns 0,1 -> StandardScaler, Columns 2,3 -> MinMaxScaler
        int[] numericCols = {0, 1};
        int[] categoricalCols = {2, 3};
        
        // Extract and scale numeric columns
        double[][] numericTrain = extractColumns(trainX, numericCols);
        double[][] numericTest = extractColumns(testX, numericCols);
        StandardScaler numericScaler = new StandardScaler();
        numericScaler.fit(numericTrain);
        double[][] scaledNumericTrain = numericScaler.transform(numericTrain);
        double[][] scaledNumericTest = numericScaler.transform(numericTest);
        
        // Extract and scale categorical columns
        double[][] catTrain = extractColumns(trainX, categoricalCols);
        double[][] catTest = extractColumns(testX, categoricalCols);
        MinMaxScaler catScaler = new MinMaxScaler();
        catScaler.fit(catTrain);
        double[][] scaledCatTrain = catScaler.transform(catTrain);
        double[][] scaledCatTest = catScaler.transform(catTest);
        
        // Concatenate
        double[][] transformedTrain = concatenate(scaledNumericTrain, scaledCatTrain);
        double[][] transformedTest = concatenate(scaledNumericTest, scaledCatTest);
        
        System.out.println("   Configuration:");
        System.out.println("     Columns [0,1] -> StandardScaler");
        System.out.println("     Columns [2,3] -> MinMaxScaler");
        System.out.println("\n   Original shape: " + trainX.length + " x " + trainX[0].length);
        System.out.println("   Transformed shape: " + transformedTrain.length + " x " + transformedTrain[0].length);
        
        // Use with classifier
        DecisionTreeClassifier clf = new DecisionTreeClassifier();
        clf.train(transformedTrain, trainY);
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
        
        System.out.println("   Searching best parameters for KNN...");
        
        // Define parameter grid
        int[] kValues = {1, 3, 5, 7, 9};
        
        System.out.println("   Parameter Grid:");
        System.out.println("     k: [1, 3, 5, 7, 9]");
        System.out.println("     Total combinations: " + kValues.length);
        
        // Manual grid search
        double bestAccuracy = 0;
        int bestK = 0;
        
        StandardScaler searchScaler = new StandardScaler();
        searchScaler.fit(trainX);
        double[][] searchScaledTrainX = searchScaler.transform(trainX);
        double[][] searchScaledTestX = searchScaler.transform(testX);
        
        System.out.println("\n   Searching...");
        for (int k : kValues) {
            KNearestNeighbors model = new KNearestNeighbors(k);
            model.train(searchScaledTrainX, trainY);
            int[] preds = model.predict(searchScaledTestX);
            
            int c = 0;
            for (int i = 0; i < testY.length; i++) {
                if (preds[i] == testY[i]) c++;
            }
            double acc = (double) c / testY.length;
            
            if (acc > bestAccuracy) {
                bestAccuracy = acc;
                bestK = k;
            }
        }
        
        System.out.println("\n   Best Parameters Found:");
        System.out.println("     k: " + bestK);
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
    
    private static double[][] extractColumns(double[][] X, int[] columns) {
        double[][] result = new double[X.length][columns.length];
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < columns.length; j++) {
                result[i][j] = X[i][columns[j]];
            }
        }
        return result;
    }
    
    private static double[][] concatenate(double[][] a, double[][] b) {
        double[][] result = new double[a.length][a[0].length + b[0].length];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, result[i], 0, a[0].length);
            System.arraycopy(b[i], 0, result[i], a[0].length, b[0].length);
        }
        return result;
    }
}
