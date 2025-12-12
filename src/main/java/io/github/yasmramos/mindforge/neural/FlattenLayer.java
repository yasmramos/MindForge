package io.github.yasmramos.mindforge.neural;

import java.io.Serializable;

/**
 * Flatten layer for Convolutional Neural Networks.
 * 
 * Reshapes multi-dimensional input to a 1D vector.
 * Typically used to transition from convolutional layers to dense layers.
 * 
 * This layer simply passes through the input as-is since the internal
 * representation in this implementation is already 1D arrays.
 * It serves as a conceptual bridge between convolutional and dense layers.
 * 
 * @author MindForge Team
 * @version 1.2.0
 */
public class FlattenLayer extends Layer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Original shape information for documentation
    private final int inputChannels;
    private final int inputHeight;
    private final int inputWidth;
    
    /**
     * Creates a flatten layer.
     *
     * @param inputChannels number of input channels
     * @param inputHeight height of input feature maps
     * @param inputWidth width of input feature maps
     */
    public FlattenLayer(int inputChannels, int inputHeight, int inputWidth) {
        if (inputChannels <= 0 || inputHeight <= 0 || inputWidth <= 0) {
            throw new IllegalArgumentException("All input dimensions must be positive");
        }
        
        this.inputChannels = inputChannels;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        
        // Both input and output sizes are the same (flattened)
        this.inputSize = inputChannels * inputHeight * inputWidth;
        this.outputSize = inputSize;
    }
    
    /**
     * Creates a flatten layer with just the total size.
     *
     * @param inputSize total size of input
     */
    public FlattenLayer(int inputSize) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive");
        }
        
        this.inputChannels = 1;
        this.inputHeight = 1;
        this.inputWidth = inputSize;
        
        this.inputSize = inputSize;
        this.outputSize = inputSize;
    }
    
    @Override
    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException(
                "Input size mismatch: expected " + inputSize + ", got " + input.length);
        }
        
        // Simply pass through (already flattened in our implementation)
        lastInput = input.clone();
        lastOutput = input.clone();
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        // No parameters to update, just pass gradient through
        return gradOutput.clone();
    }
    
    // Getters
    
    public int getInputChannels() { return inputChannels; }
    public int getInputHeight() { return inputHeight; }
    public int getInputWidth() { return inputWidth; }
    
    @Override
    public String toString() {
        return String.format("Flatten([%d, %d, %d]) -> %d",
            inputChannels, inputHeight, inputWidth, outputSize);
    }
}
