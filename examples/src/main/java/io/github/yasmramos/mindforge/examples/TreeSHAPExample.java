package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.classification.RandomForestClassifier;
import io.github.yasmramos.mindforge.interpret.TreeSHAP;
import io.github.yasmramos.mindforge.data.DatasetLoader;
import io.github.yasmramos.mindforge.validation.Metrics;

/**
 * Example demonstrating TreeSHAP for explaining Random Forest predictions.
 * 
 * TreeSHAP provides fast, exact SHAP values for tree-based models,
 * making it much faster than Kernel SHAP while maintaining accuracy.
 */
public class TreeSHAPExample {
    
    public static void main(String[] args) {
        System.out.println("=== TreeSHAP Example ===\n");
        
        // Load Iris dataset
        System.out.println("Loading Iris dataset...");
        var dataset = DatasetLoader.loadIris();
        double[][] X = dataset.getFeatures();
        int[] y = dataset.getTargets();
        
        System.out.println("Dataset: " + X.length + " samples, " + X[0].length + " features");
        System.out.println("Classes: " + dataset.getClasses().length + "\n");
        
        // Train Random Forest
        System.out.println("Training Random Forest classifier...");
        RandomForestClassifier rf = new RandomForestClassifier.Builder()
            .nEstimators(50)
            .maxDepth(10)
            .minSamplesSplit(2)
            .randomState(42)
            .build();
        
        rf.fit(X, y);
        
        // Evaluate model
        int[] predictions = rf.predict(X);
        double accuracy = Metrics.accuracy(y, predictions);
        System.out.println("Model accuracy: " + String.format("%.4f", accuracy) + "\n");
        
        // Create TreeSHAP explainer
        System.out.println("Creating TreeSHAP explainer...");
        TreeSHAP shap = new TreeSHAP(rf);
        
        // Use a subset of training data as background
        double[][] background = new double[Math.min(50, X.length)][];
        for (int i = 0; i < background.length; i++) {
            background[i] = X[i].clone();
        }
        shap.setBackground(background);
        
        System.out.println("Expected value (base prediction): " + String.format("%.4f", shap.getExpectedValue()));
        System.out.println("Model type: " + shap.getModelType());
        System.out.println("Number of features: " + shap.getNFeatures() + "\n");
        
        // Explain a single instance
        System.out.println("=== Explaining Single Instance ===");
        double[] instance = X[100]; // Pick an instance to explain
        int trueLabel = y[100];
        int predictedLabel = rf.predict(instance)[0];
        
        System.out.println("Instance: " + java.util.Arrays.toString(instance));
        System.out.println("True label: " + trueLabel);
        System.out.println("Predicted label: " + predictedLabel + "\n");
        
        // Compute SHAP values
        double[] shapValues = shap.explain(instance);
        
        System.out.println("SHAP values for each feature:");
        String[] featureNames = {"Sepal Length", "Sepal Width", "Petal Length", "Petal Width"};
        for (int i = 0; i < shapValues.length; i++) {
            System.out.println(String.format("  %s: %+.6f", featureNames[i], shapValues[i]));
        }
        
        // Verify SHAP property: sum of SHAP values + expected value = model prediction
        double shapSum = 0;
        for (double sv : shapValues) {
            shapSum += sv;
        }
        double reconstructedPrediction = shap.getExpectedValue() + shapSum;
        System.out.println("\nSHAP values sum: " + String.format("%.6f", shapSum));
        System.out.println("Expected value + SHAP sum: " + String.format("%.6f", reconstructedPrediction));
        System.out.println("(Should approximate model prediction)\n");
        
        // Compute global feature importance
        System.out.println("=== Global Feature Importance ===");
        double[] meanAbsShap = shap.meanAbsoluteShap(X);
        
        System.out.println("Mean absolute SHAP value for each feature:");
        for (int i = 0; i < meanAbsShap.length; i++) {
            System.out.println(String.format("  %s: %.6f", featureNames[i], meanAbsShap[i]));
        }
        
        // Rank features by importance
        Integer[] rankedIndices = new Integer[meanAbsShap.length];
        for (int i = 0; i < meanAbsShap.length; i++) {
            rankedIndices[i] = i;
        }
        java.util.Arrays.sort(rankedIndices, (a, b) -> Double.compare(meanAbsShap[b], meanAbsShap[a]));
        
        System.out.println("\nFeature importance ranking:");
        for (int rank = 0; rank < rankedIndices.length; rank++) {
            int idx = rankedIndices[rank];
            System.out.println(String.format("  %d. %s (%.6f)", rank + 1, featureNames[idx], meanAbsShap[idx]));
        }
        
        // Batch explanation
        System.out.println("\n=== Batch Explanation ===");
        int nSamplesToExplain = Math.min(10, X.length);
        double[][] instancesToExplain = new double[nSamplesToExplain][];
        for (int i = 0; i < nSamplesToExplain; i++) {
            instancesToExplain[i] = X[i].clone();
        }
        
        System.out.println("Explaining " + nSamplesToExplain + " instances...");
        long startTime = System.currentTimeMillis();
        double[][] allShap = shap.explainBatch(instancesToExplain);
        long endTime = System.currentTimeMillis();
        
        System.out.println("Time taken: " + (endTime - startTime) + " ms");
        System.out.println("SHAP values computed for " + allShap.length + " instances × " + allShap[0].length + " features");
        
        System.out.println("\n=== TreeSHAP Example Complete ===");
    }
}
