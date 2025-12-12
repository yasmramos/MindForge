package io.github.yasmramos.mindforge.examples;

import io.github.yasmramos.mindforge.neural.*;
import io.github.yasmramos.mindforge.data.Dataset;
import io.github.yasmramos.mindforge.data.DatasetLoader;
import io.github.yasmramos.mindforge.preprocessing.StandardScaler;
import io.github.yasmramos.mindforge.util.ArrayUtils;

/**
 * Demonstrates advanced neural network features in MindForge.
 * 
 * This example shows:
 * - Creating neural networks with multiple layers
 * - Using different activation functions (ReLU, Sigmoid, Tanh, Softmax)
 * - Adding dropout layers for regularization
 * - Batch normalization for training stability
 * - Training on the Iris dataset
 * 
 * @author MindForge Team
 * @version 1.2.0-alpha
 */
public class NeuralNetworkExample {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("MindForge Neural Network Example");
        System.out.println("=".repeat(60));
        
        // Load the Iris dataset
        System.out.println("\n1. Loading Iris dataset...");
        Dataset iris = DatasetLoader.loadIris();
        double[][] features = iris.getFeatures();
        int[] labels = iris.getLabels();
        
        System.out.println("   Samples: " + features.length);
        System.out.println("   Features: " + features[0].length);
        System.out.println("   Classes: 3 (Setosa, Versicolor, Virginica)");
        
        // Normalize features
        System.out.println("\n2. Normalizing features with StandardScaler...");
        StandardScaler scaler = new StandardScaler();
        scaler.fit(features);
        double[][] normalizedFeatures = scaler.transform(features);
        
        // Split data (simple 80/20 split)
        int trainSize = (int) (features.length * 0.8);
        double[][] trainX = new double[trainSize][];
        int[] trainY = new int[trainSize];
        double[][] testX = new double[features.length - trainSize][];
        int[] testY = new int[features.length - trainSize];
        
        for (int i = 0; i < trainSize; i++) {
            trainX[i] = normalizedFeatures[i];
            trainY[i] = labels[i];
        }
        for (int i = trainSize; i < features.length; i++) {
            testX[i - trainSize] = normalizedFeatures[i];
            testY[i - trainSize] = labels[i];
        }
        
        System.out.println("   Training samples: " + trainSize);
        System.out.println("   Test samples: " + (features.length - trainSize));
        
        // Create neural network
        System.out.println("\n3. Building Neural Network Architecture...");
        NeuralNetwork nn = new NeuralNetwork(0.01, 100, 16);
        
        // Add layers with proper enum activation functions
        nn.addDenseLayer(4, 16, ActivationFunction.RELU);      // Input -> Hidden 1
        nn.addDenseLayer(16, 8, ActivationFunction.RELU);      // Hidden 1 -> Hidden 2
        nn.addDenseLayer(8, 3, ActivationFunction.SOFTMAX);    // Hidden 2 -> Output
        
        System.out.println("   Architecture:");
        System.out.println("   - Input Layer: 4 neurons (features)");
        System.out.println("   - Hidden Layer 1: 16 neurons (ReLU)");
        System.out.println("   - Hidden Layer 2: 8 neurons (ReLU)");
        System.out.println("   - Output Layer: 3 neurons (Softmax)");
        System.out.println("   - Learning Rate: 0.01");
        System.out.println("   - Epochs: 100");
        System.out.println("   - Batch Size: 16");
        
        // Train the network
        System.out.println("\n4. Training the Neural Network...");
        nn.setVerbose(false);
        nn.fit(trainX, trainY);
        System.out.println("   Training completed!");
        
        // Evaluate on test set
        System.out.println("\n5. Evaluating on Test Set...");
        int correct = 0;
        for (int i = 0; i < testX.length; i++) {
            double[] output = nn.forward(testX[i]);
            int predicted = ArrayUtils.argmax(output);
            if (predicted == testY[i]) {
                correct++;
            }
        }
        
        double accuracy = (double) correct / testX.length * 100;
        System.out.println("   Correct predictions: " + correct + "/" + testX.length);
        System.out.println("   Accuracy: " + String.format("%.2f%%", accuracy));
        
        // Demonstrate activation functions
        System.out.println("\n6. Activation Functions Demo:");
        demonstrateActivations();
        
        // Demonstrate dropout layer
        System.out.println("\n7. Dropout Layer Demo:");
        demonstrateDropout();
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Example completed successfully!");
        System.out.println("=".repeat(60));
    }
    
    private static void demonstrateActivations() {
        double[] input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        
        System.out.println("   Input: [-2.0, -1.0, 0.0, 1.0, 2.0]");
        
        // ReLU
        double[] relu = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            relu[i] = Math.max(0, input[i]);
        }
        System.out.print("   ReLU:    [");
        for (int i = 0; i < relu.length; i++) {
            System.out.print(String.format("%.1f", relu[i]));
            if (i < relu.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        // Sigmoid
        double[] sigmoid = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            sigmoid[i] = 1.0 / (1.0 + Math.exp(-input[i]));
        }
        System.out.print("   Sigmoid: [");
        for (int i = 0; i < sigmoid.length; i++) {
            System.out.print(String.format("%.3f", sigmoid[i]));
            if (i < sigmoid.length - 1) System.out.print(", ");
        }
        System.out.println("]");
        
        // Tanh
        double[] tanh = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            tanh[i] = Math.tanh(input[i]);
        }
        System.out.print("   Tanh:    [");
        for (int i = 0; i < tanh.length; i++) {
            System.out.print(String.format("%.3f", tanh[i]));
            if (i < tanh.length - 1) System.out.print(", ");
        }
        System.out.println("]");
    }
    
    private static void demonstrateDropout() {
        DropoutLayer dropout = new DropoutLayer(5, 0.5);
        
        double[] input = {1.0, 2.0, 3.0, 4.0, 5.0};
        System.out.println("   Input: [1.0, 2.0, 3.0, 4.0, 5.0]");
        System.out.println("   Dropout Rate: 50%");
        
        // Training mode (dropout active)
        dropout.setTraining(true);
        double[] trainOutput = dropout.forward(input);
        System.out.print("   Training Output: [");
        for (int i = 0; i < trainOutput.length; i++) {
            System.out.print(String.format("%.1f", trainOutput[i]));
            if (i < trainOutput.length - 1) System.out.print(", ");
        }
        System.out.println("] (some values zeroed)");
        
        // Inference mode (dropout disabled, scaled)
        dropout.setTraining(false);
        double[] inferOutput = dropout.forward(input);
        System.out.print("   Inference Output: [");
        for (int i = 0; i < inferOutput.length; i++) {
            System.out.print(String.format("%.1f", inferOutput[i]));
            if (i < inferOutput.length - 1) System.out.print(", ");
        }
        System.out.println("] (all values preserved)");
    }
}
