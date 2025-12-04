package com.mindforge.neural;

import java.util.Random;

/**
 * Dropout layer for regularization in neural networks.
 * Randomly sets a fraction of input units to 0 during training.
 */
public class DropoutLayer extends Layer {
    private static final long serialVersionUID = 1L;
    
    private double dropoutRate;
    private boolean training;
    private boolean[] mask;
    private Random random;
    
    /**
     * Create a dropout layer.
     * 
     * @param size size of the layer (input = output size)
     * @param dropoutRate fraction of units to drop (0-1)
     */
    public DropoutLayer(int size, double dropoutRate) {
        this(size, dropoutRate, System.currentTimeMillis());
    }
    
    /**
     * Create a dropout layer with a specific seed.
     * 
     * @param size size of the layer
     * @param dropoutRate fraction of units to drop (0-1)
     * @param seed random seed
     */
    public DropoutLayer(int size, double dropoutRate, long seed) {
        this.inputSize = size;
        this.outputSize = size;
        this.dropoutRate = dropoutRate;
        this.training = true;
        this.random = new Random(seed);
    }
    
    @Override
    public double[] forward(double[] input) {
        lastInput = input.clone();
        lastOutput = new double[input.length];
        mask = new boolean[input.length];
        
        if (training) {
            double scale = 1.0 / (1.0 - dropoutRate);
            for (int i = 0; i < input.length; i++) {
                if (random.nextDouble() > dropoutRate) {
                    mask[i] = true;
                    lastOutput[i] = input[i] * scale;
                } else {
                    mask[i] = false;
                    lastOutput[i] = 0.0;
                }
            }
        } else {
            // During inference, use all units
            System.arraycopy(input, 0, lastOutput, 0, input.length);
        }
        
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        double[] gradInput = new double[inputSize];
        double scale = 1.0 / (1.0 - dropoutRate);
        
        for (int i = 0; i < inputSize; i++) {
            if (mask[i]) {
                gradInput[i] = gradOutput[i] * scale;
            }
        }
        
        return gradInput;
    }
    
    /**
     * Set training mode.
     * 
     * @param training true for training, false for inference
     */
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    /**
     * Check if in training mode.
     * 
     * @return true if training
     */
    public boolean isTraining() {
        return training;
    }
    
    /**
     * Get the dropout rate.
     * 
     * @return dropout rate
     */
    public double getDropoutRate() {
        return dropoutRate;
    }
}
