package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.pipeline.Pipeline;
import io.github.yasmramos.mindforge.preprocessing.StandardScaler;
import io.github.yasmramos.mindforge.classification.KNearestNeighbors;
import io.github.yasmramos.mindforge.classification.DecisionTreeClassifier;
import io.github.yasmramos.mindforge.classification.GaussianNaiveBayes;
import io.github.yasmramos.mindforge.preprocessing.DataSplit;
import java.util.Random;

/**
 * Pipeline Example with MindForge
 * 
 * Demonstrates ML Pipelines:
 * - Chaining preprocessing and model steps
 * - Simplified workflow (fit/predict in one object)
 * - Comparing different model pipelines
 */
public class PipelineExample {

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Pipeline Example");
        System.out.println("=".repeat(60));

        // Generate Iris-like dataset
        System.out.println("\n1. Generating classification dataset...");
        double[][] X = generateIrisLikeData();
        int[] y = generateIrisLabels();
        System.out.printf("   Created %d samples with 4 features, 3 classes%n", X.length);

        // Split data using DataSplit (testSize, shuffle, randomSeed)
        System.out.println("\n2. Splitting data (70% train, 30% test)...");
        DataSplit.TrainTestSplit split = DataSplit.trainTestSplit(X, y, 0.3, true, 42);
        System.out.printf("   Training: %d samples, Test: %d samples%n", 
                          split.XTrain.length, split.XTest.length);

        // Build Pipeline with StandardScaler + KNN
        System.out.println("\n3. Building Pipeline: StandardScaler -> KNN");
        System.out.println("   " + "-".repeat(50));
        
        Pipeline knnPipeline = new Pipeline.Builder()
            .addTransformer("scaler", new StandardScalerTransformer())
            .addClassifier("knn", new KNearestNeighbors(5))
            .build();
        
        // Fit pipeline
        knnPipeline.fit(split.XTrain, split.yTrain);
        System.out.println("   Pipeline fitted successfully!");
        
        // Predict and evaluate
        int[] predictions = knnPipeline.predict(split.XTest);
        double accuracy = knnPipeline.score(split.XTest, split.yTest);
        System.out.printf("   Pipeline accuracy: %.1f%%%n", accuracy * 100);

        // Compare multiple pipelines
        System.out.println("\n4. Comparing Different Model Pipelines:");
        System.out.println("   " + "-".repeat(50));
        
        // Pipeline 1: StandardScaler + KNN
        Pipeline pipe1 = new Pipeline.Builder()
            .addTransformer("scaler", new StandardScalerTransformer())
            .addClassifier("model", new KNearestNeighbors(3))
            .build();
        pipe1.fit(split.XTrain, split.yTrain);
        double acc1 = pipe1.score(split.XTest, split.yTest);
        
        // Pipeline 2: StandardScaler + Decision Tree
        Pipeline pipe2 = new Pipeline.Builder()
            .addTransformer("scaler", new StandardScalerTransformer())
            .addClassifier("model", new DecisionTreeClassifier())
            .build();
        pipe2.fit(split.XTrain, split.yTrain);
        double acc2 = pipe2.score(split.XTest, split.yTest);
        
        // Pipeline 3: StandardScaler + Gaussian NB
        Pipeline pipe3 = new Pipeline.Builder()
            .addTransformer("scaler", new StandardScalerTransformer())
            .addClassifier("model", new GaussianNaiveBayes())
            .build();
        pipe3.fit(split.XTrain, split.yTrain);
        double acc3 = pipe3.score(split.XTest, split.yTest);
        
        System.out.printf("   KNN (k=3):        %.1f%% accuracy%n", acc1 * 100);
        System.out.printf("   Decision Tree:    %.1f%% accuracy%n", acc2 * 100);
        System.out.printf("   Gaussian NB:      %.1f%% accuracy%n", acc3 * 100);

        // Detailed predictions
        System.out.println("\n5. Sample Predictions (first 10 test samples):");
        System.out.println("   " + "-".repeat(50));
        
        String[] classNames = {"Setosa", "Versicolor", "Virginica"};
        int[] bestPredictions = pipe1.predict(split.XTest);
        
        System.out.println("   Sample | Actual      | Predicted   | Match");
        System.out.println("   " + "-".repeat(45));
        int correct = 0;
        for (int i = 0; i < Math.min(10, split.XTest.length); i++) {
            boolean match = split.yTest[i] == bestPredictions[i];
            if (match) correct++;
            System.out.printf("   %2d     | %-11s | %-11s | %s%n",
                              i + 1,
                              classNames[split.yTest[i]],
                              classNames[bestPredictions[i]],
                              match ? "OK" : "X");
        }

        // Pipeline workflow benefits
        System.out.println("\n6. Pipeline Benefits:");
        System.out.println("   " + "-".repeat(50));
        System.out.println("   - Single object manages entire workflow");
        System.out.println("   - Prevents data leakage (fit on train only)");
        System.out.println("   - Easy to swap models for comparison");
        System.out.println("   - Reproducible preprocessing + training");
        System.out.println("   - Steps: " + knnPipeline.getStepNames());
        System.out.println("   - Total steps: " + knnPipeline.getNumSteps());

        // Summary
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Pipeline Summary:");
        System.out.println("=".repeat(60));
        System.out.println("- Pipelines chain preprocessing and models together");
        System.out.println("- Use fit() to train, predict() to classify new data");
        System.out.println("- score() computes accuracy on test data");
        System.out.println("- Always compare multiple models to find the best one!");
    }

    /**
     * StandardScaler wrapper implementing Pipeline.Transformer interface
     */
    static class StandardScalerTransformer implements Pipeline.Transformer {
        private static final long serialVersionUID = 1L;
        private StandardScaler scaler = new StandardScaler();
        
        @Override
        public void fit(double[][] X, int[] y) {
            // StandardScaler doesn't use y, but interface requires it
            scaler.fit(X);
        }
        
        @Override
        public double[][] transform(double[][] X) {
            return scaler.transform(X);
        }
        
        @Override
        public double[][] fitTransform(double[][] X, int[] y) {
            scaler.fit(X);
            return scaler.transform(X);
        }
    }

    private static double[][] generateIrisLikeData() {
        double[][] X = new double[150][4];
        Random rand = new Random(42);
        
        // Class 0: Setosa-like
        for (int i = 0; i < 50; i++) {
            X[i][0] = 5.0 + rand.nextGaussian() * 0.35;  // sepal length
            X[i][1] = 3.4 + rand.nextGaussian() * 0.38;  // sepal width
            X[i][2] = 1.5 + rand.nextGaussian() * 0.17;  // petal length
            X[i][3] = 0.2 + rand.nextGaussian() * 0.10;  // petal width
        }
        
        // Class 1: Versicolor-like
        for (int i = 50; i < 100; i++) {
            X[i][0] = 5.9 + rand.nextGaussian() * 0.52;
            X[i][1] = 2.8 + rand.nextGaussian() * 0.31;
            X[i][2] = 4.3 + rand.nextGaussian() * 0.47;
            X[i][3] = 1.3 + rand.nextGaussian() * 0.20;
        }
        
        // Class 2: Virginica-like
        for (int i = 100; i < 150; i++) {
            X[i][0] = 6.6 + rand.nextGaussian() * 0.64;
            X[i][1] = 3.0 + rand.nextGaussian() * 0.32;
            X[i][2] = 5.5 + rand.nextGaussian() * 0.55;
            X[i][3] = 2.0 + rand.nextGaussian() * 0.27;
        }
        
        return X;
    }

    private static int[] generateIrisLabels() {
        int[] y = new int[150];
        for (int i = 0; i < 50; i++) y[i] = 0;
        for (int i = 50; i < 100; i++) y[i] = 1;
        for (int i = 100; i < 150; i++) y[i] = 2;
        return y;
    }
}
