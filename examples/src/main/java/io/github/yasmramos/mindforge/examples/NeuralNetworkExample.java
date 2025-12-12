package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.data.Dataset;
import io.github.yasmramos.mindforge.data.DatasetLoader;
import io.github.yasmramos.mindforge.neural.*;
import io.github.yasmramos.mindforge.validation.ConfusionMatrix;
import io.github.yasmramos.mindforge.validation.Metrics;

/**
 * Example demonstrating neural network usage with MindForge.
 */
public class NeuralNetworkExample {
    
    public static void main(String[] args) {
        System.out.println("=== MindForge Neural Network Example ===\n");
        
        // Load the Iris dataset
        Dataset iris = DatasetLoader.loadIris();
        System.out.println("Dataset: " + iris);
        System.out.println("Features: " + String.join(", ", iris.getFeatureNames()));
        System.out.println("Classes: " + String.join(", ", iris.getTargetNames()));
        
        // Split into training and test sets
        Dataset[] splits = iris.trainTestSplit(0.2, 42);
        Dataset trainSet = splits[0];
        Dataset testSet = splits[1];
        
        System.out.println("\nTraining samples: " + trainSet.getNumSamples());
        System.out.println("Test samples: " + testSet.getNumSamples());
        
        // Create neural network
        System.out.println("\n--- Creating Neural Network ---");
        NeuralNetwork nn = new NeuralNetwork(0.01, 100, 16);
        
        // Add layers: 4 input -> 16 hidden (ReLU) -> 8 hidden (ReLU) -> 3 output (Softmax)
        nn.addDenseLayer(4, 16, ActivationFunction.RELU);
        nn.addDenseLayer(16, 8, ActivationFunction.RELU);
        nn.addDenseLayer(8, 3, ActivationFunction.SOFTMAX);
        
        System.out.println("Network architecture:");
        System.out.println("  Input: 4 features");
        System.out.println("  Hidden Layer 1: 16 neurons (ReLU)");
        System.out.println("  Hidden Layer 2: 8 neurons (ReLU)");
        System.out.println("  Output: 3 classes (Softmax)");
        
        // Train the network
        System.out.println("\n--- Training ---");
        nn.setVerbose(true);
        nn.fit(trainSet.getFeatures(), trainSet.getLabels(),
               testSet.getFeatures(), testSet.getLabels());
        
        // Evaluate on test set
        System.out.println("\n--- Evaluation ---");
        int[] predictions = nn.predict(testSet.getFeatures());
        int[] trueLabels = testSet.getLabels();
        
        double accuracy = Metrics.accuracy(trueLabels, predictions);
        System.out.println("Test Accuracy: " + String.format("%.4f", accuracy));
        
        // Confusion matrix
        ConfusionMatrix cm = new ConfusionMatrix(trueLabels, predictions, iris.getTargetNames());
        System.out.println("\n" + cm.toString());
        
        // Classification report
        System.out.println("Classification Report:");
        System.out.println(cm.classificationReport());
        
        // Demonstrate XOR problem (non-linear)
        System.out.println("\n=== XOR Problem (Non-linear) ===\n");
        demonstrateXOR();
    }
    
    private static void demonstrateXOR() {
        // Generate XOR dataset
        Dataset xor = DatasetLoader.makeXOR(50, 0.2, 42);
        System.out.println("Dataset: " + xor);
        
        Dataset[] splits = xor.trainTestSplit(0.2, 42);
        Dataset trainSet = splits[0];
        Dataset testSet = splits[1];
        
        // Create a deeper network for XOR
        NeuralNetwork nn = new NeuralNetwork(0.1, 200, 32);
        nn.addDenseLayer(2, 8, ActivationFunction.TANH);
        nn.addDenseLayer(8, 4, ActivationFunction.TANH);
        nn.addDenseLayer(4, 2, ActivationFunction.SOFTMAX);
        
        System.out.println("Network: 2 -> 8 (tanh) -> 4 (tanh) -> 2 (softmax)");
        
        // Train
        nn.setVerbose(false);
        nn.fit(trainSet.getFeatures(), trainSet.getLabels());
        
        // Evaluate
        int[] predictions = nn.predict(testSet.getFeatures());
        double accuracy = Metrics.accuracy(testSet.getLabels(), predictions);
        
        System.out.println("Test Accuracy: " + String.format("%.4f", accuracy));
        System.out.println("(XOR is non-linearly separable, requires hidden layers)");
    }
}
