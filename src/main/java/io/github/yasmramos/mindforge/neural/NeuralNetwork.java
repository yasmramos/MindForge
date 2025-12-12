package io.github.yasmramos.mindforge.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Multilayer Perceptron (MLP) Neural Network.
 * Supports multiple hidden layers with various activation functions.
 */
public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private List<Layer> layers;
    private double learningRate;
    private int epochs;
    private int batchSize;
    private boolean verbose;
    private List<Double> trainingLoss;
    private List<Double> validationLoss;
    
    /**
     * Create a neural network with default parameters.
     */
    public NeuralNetwork() {
        this(0.01, 100, 32);
    }
    
    /**
     * Create a neural network with specified parameters.
     * 
     * @param learningRate learning rate for optimization
     * @param epochs number of training epochs
     * @param batchSize mini-batch size
     */
    public NeuralNetwork(double learningRate, int epochs, int batchSize) {
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.verbose = true;
        this.trainingLoss = new ArrayList<>();
        this.validationLoss = new ArrayList<>();
    }
    
    /**
     * Add a layer to the network.
     * 
     * @param layer layer to add
     * @return this network for chaining
     */
    public NeuralNetwork addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }
    
    /**
     * Add a dense layer to the network.
     * 
     * @param inputSize input size (only needed for first layer)
     * @param outputSize number of neurons
     * @param activation activation function
     * @return this network for chaining
     */
    public NeuralNetwork addDenseLayer(int inputSize, int outputSize, ActivationFunction activation) {
        layers.add(new DenseLayer(inputSize, outputSize, activation));
        return this;
    }
    
    /**
     * Forward pass through the network.
     * 
     * @param input input features
     * @return output predictions
     */
    public double[] forward(double[] input) {
        double[] current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
    
    /**
     * Backward pass through the network.
     * 
     * @param gradOutput gradient of the loss
     */
    private void backward(double[] gradOutput) {
        double[] grad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad, learningRate);
        }
    }
    
    /**
     * Train the network for classification.
     * 
     * @param X training features
     * @param y training labels (class indices)
     */
    public void fit(double[][] X, int[] y) {
        fit(X, y, null, null);
    }
    
    /**
     * Train the network for classification with validation data.
     * 
     * @param X training features
     * @param y training labels
     * @param XVal validation features
     * @param yVal validation labels
     */
    public void fit(double[][] X, int[] y, double[][] XVal, int[] yVal) {
        int nSamples = X.length;
        int nClasses = 0;
        for (int label : y) {
            if (label >= nClasses) nClasses = label + 1;
        }
        
        // Convert labels to one-hot encoding
        double[][] yOneHot = new double[nSamples][nClasses];
        for (int i = 0; i < nSamples; i++) {
            yOneHot[i][y[i]] = 1.0;
        }
        
        Random random = new Random(42);
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle indices
            for (int i = nSamples - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            double epochLoss = 0.0;
            
            // Mini-batch training
            for (int batch = 0; batch < nSamples; batch += batchSize) {
                int batchEnd = Math.min(batch + batchSize, nSamples);
                
                for (int i = batch; i < batchEnd; i++) {
                    int idx = indices[i];
                    
                    // Forward pass
                    double[] output = forward(X[idx]);
                    
                    // Calculate cross-entropy loss gradient
                    double[] gradOutput = new double[output.length];
                    for (int j = 0; j < output.length; j++) {
                        gradOutput[j] = output[j] - yOneHot[idx][j];
                        epochLoss -= yOneHot[idx][j] * Math.log(Math.max(output[j], 1e-15));
                    }
                    
                    // Backward pass
                    backward(gradOutput);
                }
            }
            
            epochLoss /= nSamples;
            trainingLoss.add(epochLoss);
            
            // Validation loss
            if (XVal != null && yVal != null) {
                double valLoss = calculateLoss(XVal, yVal);
                validationLoss.add(valLoss);
                
                if (verbose && epoch % 10 == 0) {
                    System.out.printf("Epoch %d/%d - Loss: %.4f - Val Loss: %.4f%n",
                            epoch + 1, epochs, epochLoss, valLoss);
                }
            } else if (verbose && epoch % 10 == 0) {
                System.out.printf("Epoch %d/%d - Loss: %.4f%n", epoch + 1, epochs, epochLoss);
            }
        }
    }
    
    /**
     * Train the network for regression.
     * 
     * @param X training features
     * @param y training targets
     */
    public void fitRegression(double[][] X, double[] y) {
        fitRegression(X, y, null, null);
    }
    
    /**
     * Train the network for regression with validation data.
     * 
     * @param X training features
     * @param y training targets
     * @param XVal validation features
     * @param yVal validation targets
     */
    public void fitRegression(double[][] X, double[] y, double[][] XVal, double[] yVal) {
        int nSamples = X.length;
        
        Random random = new Random(42);
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) indices[i] = i;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle indices
            for (int i = nSamples - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
            
            double epochLoss = 0.0;
            
            for (int batch = 0; batch < nSamples; batch += batchSize) {
                int batchEnd = Math.min(batch + batchSize, nSamples);
                
                for (int i = batch; i < batchEnd; i++) {
                    int idx = indices[i];
                    
                    // Forward pass
                    double[] output = forward(X[idx]);
                    
                    // MSE loss gradient
                    double[] gradOutput = new double[output.length];
                    for (int j = 0; j < output.length; j++) {
                        double error = output[j] - y[idx];
                        gradOutput[j] = 2 * error / output.length;
                        epochLoss += error * error;
                    }
                    
                    // Backward pass
                    backward(gradOutput);
                }
            }
            
            epochLoss /= nSamples;
            trainingLoss.add(epochLoss);
            
            if (verbose && epoch % 10 == 0) {
                System.out.printf("Epoch %d/%d - MSE: %.4f%n", epoch + 1, epochs, epochLoss);
            }
        }
    }
    
    /**
     * Predict class labels.
     * 
     * @param X features
     * @return predicted class labels
     */
    public int[] predict(double[][] X) {
        int[] predictions = new int[X.length];
        for (int i = 0; i < X.length; i++) {
            double[] output = forward(X[i]);
            predictions[i] = argmax(output);
        }
        return predictions;
    }
    
    /**
     * Predict class probabilities.
     * 
     * @param X features
     * @return predicted probabilities for each class
     */
    public double[][] predictProba(double[][] X) {
        double[][] probabilities = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = forward(X[i]);
        }
        return probabilities;
    }
    
    /**
     * Predict regression values.
     * 
     * @param X features
     * @return predicted values
     */
    public double[] predictRegression(double[][] X) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double[] output = forward(X[i]);
            predictions[i] = output[0];
        }
        return predictions;
    }
    
    /**
     * Calculate the classification accuracy.
     * 
     * @param X features
     * @param y true labels
     * @return accuracy
     */
    public double score(double[][] X, int[] y) {
        int[] predictions = predict(X);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            if (predictions[i] == y[i]) correct++;
        }
        return (double) correct / y.length;
    }
    
    private double calculateLoss(double[][] X, int[] y) {
        double loss = 0.0;
        int nClasses = 0;
        for (int label : y) {
            if (label >= nClasses) nClasses = label + 1;
        }
        
        for (int i = 0; i < X.length; i++) {
            double[] output = forward(X[i]);
            loss -= Math.log(Math.max(output[y[i]], 1e-15));
        }
        return loss / X.length;
    }
    
    private int argmax(double[] array) {
        int maxIdx = 0;
        double maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Set verbose mode.
     * 
     * @param verbose whether to print training progress
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
    
    /**
     * Get training loss history.
     * 
     * @return list of training losses per epoch
     */
    public List<Double> getTrainingLoss() {
        return trainingLoss;
    }
    
    /**
     * Get validation loss history.
     * 
     * @return list of validation losses per epoch
     */
    public List<Double> getValidationLoss() {
        return validationLoss;
    }
    
    /**
     * Get the layers.
     * 
     * @return list of layers
     */
    public List<Layer> getLayers() {
        return layers;
    }
    
    /**
     * Set learning rate.
     * 
     * @param learningRate new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Set number of epochs.
     * 
     * @param epochs number of training epochs
     */
    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
    
    /**
     * Set batch size.
     * 
     * @param batchSize mini-batch size
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
