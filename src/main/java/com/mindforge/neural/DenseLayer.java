package com.mindforge.neural;

import java.util.Random;

/**
 * Fully connected (dense) layer for neural networks.
 */
public class DenseLayer extends Layer {
    private static final long serialVersionUID = 1L;
    
    private double[][] weights;
    private double[] biases;
    private ActivationFunction activation;
    private double[] preActivation;
    
    // For momentum-based optimizers
    private double[][] weightMomentum;
    private double[] biasMomentum;
    private double momentum = 0.9;
    
    /**
     * Create a dense layer with random initialization.
     * 
     * @param inputSize number of input neurons
     * @param outputSize number of output neurons
     * @param activation activation function
     */
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation) {
        this(inputSize, outputSize, activation, System.currentTimeMillis());
    }
    
    /**
     * Create a dense layer with seeded random initialization.
     * 
     * @param inputSize number of input neurons
     * @param outputSize number of output neurons
     * @param activation activation function
     * @param seed random seed for reproducibility
     */
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activation, long seed) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        
        Random random = new Random(seed);
        
        // Xavier/Glorot initialization
        double scale = Math.sqrt(2.0 / (inputSize + outputSize));
        
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        weightMomentum = new double[outputSize][inputSize];
        biasMomentum = new double[outputSize];
        
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextGaussian() * scale;
            }
            biases[i] = 0.0;
        }
    }
    
    @Override
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + input.length);
        }
        
        lastInput = input.clone();
        preActivation = new double[outputSize];
        lastOutput = new double[outputSize];
        
        for (int i = 0; i < outputSize; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputSize; j++) {
                sum += weights[i][j] * input[j];
            }
            preActivation[i] = sum;
            lastOutput[i] = activation.apply(sum);
        }
        
        // Handle softmax separately
        if (activation == ActivationFunction.SOFTMAX) {
            lastOutput = ActivationFunction.softmax(preActivation);
        }
        
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        double[] gradInput = new double[inputSize];
        
        for (int i = 0; i < outputSize; i++) {
            double gradActivation;
            if (activation == ActivationFunction.SOFTMAX) {
                // For softmax with cross-entropy, gradient is simplified
                gradActivation = gradOutput[i];
            } else {
                gradActivation = gradOutput[i] * activation.derivative(preActivation[i]);
            }
            
            // Gradient with respect to input
            for (int j = 0; j < inputSize; j++) {
                gradInput[j] += weights[i][j] * gradActivation;
            }
            
            // Update weights and biases with momentum
            for (int j = 0; j < inputSize; j++) {
                double gradWeight = gradActivation * lastInput[j];
                weightMomentum[i][j] = momentum * weightMomentum[i][j] + learningRate * gradWeight;
                weights[i][j] -= weightMomentum[i][j];
            }
            
            biasMomentum[i] = momentum * biasMomentum[i] + learningRate * gradActivation;
            biases[i] -= biasMomentum[i];
        }
        
        return gradInput;
    }
    
    /**
     * Get the weights matrix.
     * 
     * @return weights
     */
    public double[][] getWeights() {
        return weights;
    }
    
    /**
     * Get the biases.
     * 
     * @return biases
     */
    public double[] getBiases() {
        return biases;
    }
    
    /**
     * Set the momentum factor for SGD with momentum.
     * 
     * @param momentum momentum value (0-1)
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }
    
    /**
     * Get the activation function.
     * 
     * @return activation function
     */
    public ActivationFunction getActivation() {
        return activation;
    }
}
