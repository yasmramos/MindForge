package com.mindforge.examples;

import com.mindforge.data.Dataset;
import com.mindforge.data.DatasetLoader;
import com.mindforge.neural.*;
import com.mindforge.validation.ConfusionMatrix;
import com.mindforge.validation.ROCCurve;
import com.mindforge.visualization.ChartGenerator;

import java.io.IOException;
import java.util.List;

/**
 * Example demonstrating visualization capabilities.
 */
public class VisualizationExample {
    
    public static void main(String[] args) {
        System.out.println("=== MindForge Visualization Example ===\n");
        
        try {
            // Create output directory
            java.io.File outputDir = new java.io.File("charts");
            outputDir.mkdirs();
            
            // Example 1: Training History Chart
            System.out.println("1. Generating training history chart...");
            generateTrainingHistoryChart();
            
            // Example 2: Confusion Matrix Heatmap
            System.out.println("2. Generating confusion matrix heatmap...");
            generateConfusionMatrixChart();
            
            // Example 3: ROC Curve
            System.out.println("3. Generating ROC curve...");
            generateROCCurveChart();
            
            // Example 4: Scatter Plot
            System.out.println("4. Generating scatter plot...");
            generateScatterPlot();
            
            // Example 5: Bar Chart
            System.out.println("5. Generating bar chart...");
            generateBarChart();
            
            System.out.println("\nAll charts generated in 'charts/' directory!");
            System.out.println("Open the HTML files in a browser to view.");
            
        } catch (IOException e) {
            System.err.println("Error generating charts: " + e.getMessage());
        }
    }
    
    private static void generateTrainingHistoryChart() throws IOException {
        // Train a neural network and capture loss history
        Dataset iris = DatasetLoader.loadIris();
        Dataset[] splits = iris.trainTestSplit(0.2, 42);
        
        NeuralNetwork nn = new NeuralNetwork(0.01, 50, 16);
        nn.addDenseLayer(4, 8, ActivationFunction.RELU);
        nn.addDenseLayer(8, 3, ActivationFunction.SOFTMAX);
        nn.setVerbose(false);
        
        nn.fit(splits[0].getFeatures(), splits[0].getLabels(),
               splits[1].getFeatures(), splits[1].getLabels());
        
        // Generate chart
        ChartGenerator.trainingHistory(nn.getTrainingLoss(), nn.getValidationLoss(),
                "charts/training_history.html");
        System.out.println("   -> charts/training_history.html");
    }
    
    private static void generateConfusionMatrixChart() throws IOException {
        // Train a model and get predictions
        Dataset iris = DatasetLoader.loadIris();
        Dataset[] splits = iris.trainTestSplit(0.2, 42);
        
        NeuralNetwork nn = new NeuralNetwork(0.01, 100, 16);
        nn.addDenseLayer(4, 8, ActivationFunction.RELU);
        nn.addDenseLayer(8, 3, ActivationFunction.SOFTMAX);
        nn.setVerbose(false);
        nn.fit(splits[0].getFeatures(), splits[0].getLabels());
        
        int[] predictions = nn.predict(splits[1].getFeatures());
        ConfusionMatrix cm = new ConfusionMatrix(splits[1].getLabels(), predictions);
        
        // Generate heatmap
        ChartGenerator.confusionMatrixHeatmap(cm.getMatrix(), iris.getTargetNames(),
                "charts/confusion_matrix.html");
        System.out.println("   -> charts/confusion_matrix.html");
    }
    
    private static void generateROCCurveChart() throws IOException {
        // Create binary classification problem
        Dataset cancer = DatasetLoader.loadBreastCancer();
        Dataset[] splits = cancer.trainTestSplit(0.3, 42);
        
        NeuralNetwork nn = new NeuralNetwork(0.01, 100, 8);
        nn.addDenseLayer(10, 8, ActivationFunction.RELU);
        nn.addDenseLayer(8, 2, ActivationFunction.SOFTMAX);
        nn.setVerbose(false);
        nn.fit(splits[0].getFeatures(), splits[0].getLabels());
        
        // Get probabilities for positive class
        double[][] probas = nn.predictProba(splits[1].getFeatures());
        double[] positiveProba = new double[probas.length];
        for (int i = 0; i < probas.length; i++) {
            positiveProba[i] = probas[i][1];
        }
        
        // Calculate ROC curve
        ROCCurve roc = new ROCCurve(splits[1].getLabels(), positiveProba);
        
        // Generate chart
        ChartGenerator.rocCurve(roc.getFPR(), roc.getTPR(), roc.getAUC(),
                "charts/roc_curve.html");
        System.out.println("   -> charts/roc_curve.html (AUC = " + 
                String.format("%.4f", roc.getAUC()) + ")");
    }
    
    private static void generateScatterPlot() throws IOException {
        // Generate circles dataset for visualization
        Dataset circles = DatasetLoader.makeCircles(100, 0.1, 42);
        
        double[] x = new double[circles.getNumSamples()];
        double[] y = new double[circles.getNumSamples()];
        
        for (int i = 0; i < circles.getNumSamples(); i++) {
            x[i] = circles.getFeatures()[i][0];
            y[i] = circles.getFeatures()[i][1];
        }
        
        ChartGenerator.scatterPlot("Circles Dataset", "X", "Y",
                x, y, circles.getLabels(), "charts/scatter_plot.html");
        System.out.println("   -> charts/scatter_plot.html");
    }
    
    private static void generateBarChart() throws IOException {
        // Model comparison results
        String[] models = {"Random Forest", "Neural Network", "KNN", "SVM", "Naive Bayes"};
        double[] accuracies = {0.956, 0.943, 0.921, 0.934, 0.912};
        
        ChartGenerator.barChart("Model Accuracy Comparison", models, accuracies,
                "charts/model_comparison.html");
        System.out.println("   -> charts/model_comparison.html");
    }
}
