package io.github.yasmramos.mindforge.neural;

/**
 * Batch Normalization layer for neural networks.
 * Normalizes the inputs to have zero mean and unit variance.
 */
public class BatchNormLayer extends Layer {
    private static final long serialVersionUID = 1L;
    
    private double[] gamma;  // Scale parameter
    private double[] beta;   // Shift parameter
    private double[] runningMean;
    private double[] runningVar;
    private double momentum;
    private double epsilon;
    private boolean training;
    
    // For backward pass
    private double[] normalizedInput;
    private double[] inputMean;
    private double[] inputVar;
    
    /**
     * Create a batch normalization layer.
     * 
     * @param size size of the layer
     */
    public BatchNormLayer(int size) {
        this(size, 0.1, 1e-5);
    }
    
    /**
     * Create a batch normalization layer with specific parameters.
     * 
     * @param size size of the layer
     * @param momentum momentum for running statistics
     * @param epsilon small constant for numerical stability
     */
    public BatchNormLayer(int size, double momentum, double epsilon) {
        this.inputSize = size;
        this.outputSize = size;
        this.momentum = momentum;
        this.epsilon = epsilon;
        this.training = true;
        
        // Initialize parameters
        gamma = new double[size];
        beta = new double[size];
        runningMean = new double[size];
        runningVar = new double[size];
        
        for (int i = 0; i < size; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
            runningVar[i] = 1.0;
        }
    }
    
    @Override
    public double[] forward(double[] input) {
        lastInput = input.clone();
        lastOutput = new double[outputSize];
        normalizedInput = new double[outputSize];
        
        if (training) {
            // Calculate batch statistics (for single sample, use running stats)
            inputMean = new double[outputSize];
            inputVar = new double[outputSize];
            
            for (int i = 0; i < outputSize; i++) {
                inputMean[i] = input[i];
                inputVar[i] = epsilon;  // For single sample
                
                // Update running statistics
                runningMean[i] = (1 - momentum) * runningMean[i] + momentum * inputMean[i];
                runningVar[i] = (1 - momentum) * runningVar[i] + momentum * inputVar[i];
            }
        } else {
            inputMean = runningMean;
            inputVar = runningVar;
        }
        
        // Normalize and scale
        for (int i = 0; i < outputSize; i++) {
            normalizedInput[i] = (input[i] - inputMean[i]) / Math.sqrt(inputVar[i] + epsilon);
            lastOutput[i] = gamma[i] * normalizedInput[i] + beta[i];
        }
        
        return lastOutput;
    }
    
    @Override
    public double[] backward(double[] gradOutput, double learningRate) {
        double[] gradInput = new double[inputSize];
        
        for (int i = 0; i < inputSize; i++) {
            // Gradient with respect to gamma and beta
            double gradGamma = gradOutput[i] * normalizedInput[i];
            double gradBeta = gradOutput[i];
            
            // Gradient with respect to normalized input
            double gradNorm = gradOutput[i] * gamma[i];
            
            // Gradient with respect to input (simplified for single sample)
            gradInput[i] = gradNorm / Math.sqrt(inputVar[i] + epsilon);
            
            // Update parameters
            gamma[i] -= learningRate * gradGamma;
            beta[i] -= learningRate * gradBeta;
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
     * Get gamma (scale) parameters.
     * 
     * @return gamma values
     */
    public double[] getGamma() {
        return gamma;
    }
    
    /**
     * Get beta (shift) parameters.
     * 
     * @return beta values
     */
    public double[] getBeta() {
        return beta;
    }
}
