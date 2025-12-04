package com.mindforge.neural;

import java.io.Serializable;

/**
 * Abstract base class for neural network layers.
 */
public abstract class Layer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    protected int inputSize;
    protected int outputSize;
    protected double[] lastInput;
    protected double[] lastOutput;
    
    /**
     * Forward pass through the layer.
     * 
     * @param input input values
     * @return output values
     */
    public abstract double[] forward(double[] input);
    
    /**
     * Backward pass through the layer.
     * 
     * @param gradOutput gradient from the next layer
     * @param learningRate learning rate for weight updates
     * @return gradient to pass to the previous layer
     */
    public abstract double[] backward(double[] gradOutput, double learningRate);
    
    /**
     * Get the number of input neurons.
     * 
     * @return input size
     */
    public int getInputSize() {
        return inputSize;
    }
    
    /**
     * Get the number of output neurons.
     * 
     * @return output size
     */
    public int getOutputSize() {
        return outputSize;
    }
    
    /**
     * Get the last output of this layer.
     * 
     * @return last output values
     */
    public double[] getLastOutput() {
        return lastOutput;
    }
}
