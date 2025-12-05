package com.mindforge.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * LSTM Layer implementation.
 * 
 * This layer extends the base Layer class and provides LSTM functionality
 * for processing sequential data. It wraps LSTMCell and handles the
 * unrolling of the network through time.
 * 
 * Features:
 * - Bidirectional support (optional)
 * - Return sequences or last output only
 * - Stateful mode for processing very long sequences
 * - Dropout support for regularization
 * 
 * @author MindForge
 * @version 1.0
 */
public class LSTMLayer extends Layer implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // LSTM cells
    private LSTMCell forwardCell;
    private LSTMCell backwardCell;  // For bidirectional mode
    
    // Configuration
    private final int lstmInputSize;
    private final int hiddenSize;
    private final boolean returnSequences;
    private final boolean bidirectional;
    private final boolean stateful;
    private final double dropout;
    
    // State management
    private double[] currentHiddenState;
    private double[] currentCellState;
    private double[] backwardHiddenState;
    private double[] backwardCellState;
    
    // Cache for backward pass
    private List<double[]> cachedInputs;
    private List<double[]> cachedHiddenStates;
    private List<double[]> cachedCellStates;
    private List<double[]> backwardCachedHiddenStates;
    private List<double[]> backwardCachedCellStates;
    
    // Dropout mask
    private double[][] dropoutMask;
    private boolean training = true;
    
    /**
     * Creates a new LSTM layer with default settings.
     * 
     * @param inputSize Size of input features
     * @param hiddenSize Size of hidden state
     */
    public LSTMLayer(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false, false, false, 0.0);
    }
    
    /**
     * Creates a new LSTM layer with full configuration.
     * 
     * @param inputSize Size of input features
     * @param hiddenSize Size of hidden state
     * @param returnSequences Whether to return full sequence or just last output
     * @param bidirectional Whether to use bidirectional processing
     * @param stateful Whether to maintain state between batches
     * @param dropout Dropout rate (0.0 to 1.0)
     */
    public LSTMLayer(int inputSize, int hiddenSize, boolean returnSequences, 
                     boolean bidirectional, boolean stateful, double dropout) {
        this.lstmInputSize = inputSize;
        this.inputSize = inputSize;  // For Layer base class
        this.outputSize = bidirectional ? hiddenSize * 2 : hiddenSize;  // For Layer base class
        this.hiddenSize = hiddenSize;
        this.returnSequences = returnSequences;
        this.bidirectional = bidirectional;
        this.stateful = stateful;
        this.dropout = dropout;
        
        // Initialize forward cell
        this.forwardCell = new LSTMCell(inputSize, hiddenSize);
        
        // Initialize backward cell for bidirectional mode
        if (bidirectional) {
            this.backwardCell = new LSTMCell(inputSize, hiddenSize);
        }
        
        // Initialize states
        resetStates();
    }
    
    /**
     * Reset the internal states to zeros.
     */
    public void resetStates() {
        currentHiddenState = new double[hiddenSize];
        currentCellState = new double[hiddenSize];
        
        if (bidirectional) {
            backwardHiddenState = new double[hiddenSize];
            backwardCellState = new double[hiddenSize];
        }
    }
    
    /**
     * Forward pass through the LSTM layer.
     * 
     * @param input 2D array of shape [sequenceLength, inputSize]
     * @return Output array: [sequenceLength, hiddenSize(*2 if bidirectional)] if returnSequences,
     *         else [1, hiddenSize(*2 if bidirectional)]
     */
    @Override
    public double[] forward(double[] input) {
        this.lastInput = input.clone();
        // Reshape 1D input to 2D sequence
        int sequenceLength = input.length / lstmInputSize;
        double[][] sequence = new double[sequenceLength][lstmInputSize];
        for (int t = 0; t < sequenceLength; t++) {
            System.arraycopy(input, t * lstmInputSize, sequence[t], 0, lstmInputSize);
        }
        
        double[][] output = forwardSequence(sequence);
        
        // Flatten output back to 1D
        int flatOutputSize = output.length * output[0].length;
        double[] result = new double[flatOutputSize];
        for (int t = 0; t < output.length; t++) {
            System.arraycopy(output[t], 0, result, t * output[0].length, output[0].length);
        }
        
        this.lastOutput = result;
        return result;
    }
    
    /**
     * Forward pass for 2D sequence input.
     * 
     * @param sequence 2D array of shape [sequenceLength, inputSize]
     * @return Output sequence
     */
    public double[][] forwardSequence(double[][] sequence) {
        int sequenceLength = sequence.length;
        
        // Initialize cache
        cachedInputs = new ArrayList<>();
        cachedHiddenStates = new ArrayList<>();
        cachedCellStates = new ArrayList<>();
        
        if (bidirectional) {
            backwardCachedHiddenStates = new ArrayList<>();
            backwardCachedCellStates = new ArrayList<>();
        }
        
        // Initialize states if not stateful
        if (!stateful) {
            resetStates();
        }
        
        // Generate dropout mask if training
        int outputDim = bidirectional ? hiddenSize * 2 : hiddenSize;
        if (training && dropout > 0) {
            dropoutMask = new double[sequenceLength][outputDim];
            for (int t = 0; t < sequenceLength; t++) {
                for (int i = 0; i < outputDim; i++) {
                    dropoutMask[t][i] = Math.random() > dropout ? 1.0 / (1.0 - dropout) : 0.0;
                }
            }
        }
        
        // Forward pass through time
        double[][] forwardOutputs = new double[sequenceLength][hiddenSize];
        double[] h = currentHiddenState.clone();
        double[] c = currentCellState.clone();
        
        cachedHiddenStates.add(h.clone());
        cachedCellStates.add(c.clone());
        
        for (int t = 0; t < sequenceLength; t++) {
            cachedInputs.add(sequence[t].clone());
            
            double[][] result = forwardCell.forward(sequence[t], h, c);
            h = result[0];
            c = result[1];
            
            forwardOutputs[t] = h.clone();
            cachedHiddenStates.add(h.clone());
            cachedCellStates.add(c.clone());
        }
        
        // Update current states
        currentHiddenState = h;
        currentCellState = c;
        
        // Backward pass for bidirectional
        double[][] backwardOutputs = null;
        if (bidirectional) {
            backwardOutputs = new double[sequenceLength][hiddenSize];
            h = backwardHiddenState.clone();
            c = backwardCellState.clone();
            
            backwardCachedHiddenStates.add(h.clone());
            backwardCachedCellStates.add(c.clone());
            
            for (int t = sequenceLength - 1; t >= 0; t--) {
                double[][] result = backwardCell.forward(sequence[t], h, c);
                h = result[0];
                c = result[1];
                
                backwardOutputs[t] = h.clone();
                backwardCachedHiddenStates.add(0, h.clone());
                backwardCachedCellStates.add(0, c.clone());
            }
            
            backwardHiddenState = h;
            backwardCellState = c;
        }
        
        // Combine outputs
        double[][] output;
        if (returnSequences) {
            output = new double[sequenceLength][outputDim];
            for (int t = 0; t < sequenceLength; t++) {
                System.arraycopy(forwardOutputs[t], 0, output[t], 0, hiddenSize);
                if (bidirectional) {
                    System.arraycopy(backwardOutputs[t], 0, output[t], hiddenSize, hiddenSize);
                }
            }
        } else {
            output = new double[1][outputDim];
            System.arraycopy(forwardOutputs[sequenceLength - 1], 0, output[0], 0, hiddenSize);
            if (bidirectional) {
                System.arraycopy(backwardOutputs[0], 0, output[0], hiddenSize, hiddenSize);
            }
        }
        
        // Apply dropout
        if (training && dropout > 0) {
            for (int t = 0; t < output.length; t++) {
                for (int i = 0; i < output[0].length; i++) {
                    output[t][i] *= dropoutMask[t % dropoutMask.length][i];
                }
            }
        }
        
        return output;
    }
    
    /**
     * Backward pass through the LSTM layer.
     * 
     * @param outputGradient Gradient from the next layer
     * @param learningRate Learning rate for weight updates
     * @return Gradient with respect to the input
     */
    @Override
    public double[] backward(double[] outputGradient, double learningRate) {
        // Reshape 1D gradient to 2D
        int outputDim = bidirectional ? hiddenSize * 2 : hiddenSize;
        int gradLength = returnSequences ? cachedInputs.size() : 1;
        double[][] gradSequence = new double[gradLength][outputDim];
        
        for (int t = 0; t < gradLength; t++) {
            System.arraycopy(outputGradient, t * outputDim, gradSequence[t], 0, outputDim);
        }
        
        double[][] inputGradient = backwardSequence(gradSequence);
        
        // Update weights using learning rate
        forwardCell.setLearningRate(learningRate);
        if (bidirectional) {
            backwardCell.setLearningRate(learningRate);
        }
        updateWeights(1);
        
        // Flatten input gradient
        int inputGradSize = inputGradient.length * lstmInputSize;
        double[] result = new double[inputGradSize];
        for (int t = 0; t < inputGradient.length; t++) {
            System.arraycopy(inputGradient[t], 0, result, t * lstmInputSize, lstmInputSize);
        }
        
        return result;
    }
    
    /**
     * Backward pass for 2D sequence gradient.
     * 
     * @param outputGradient 2D gradient array
     * @return Input gradient
     */
    public double[][] backwardSequence(double[][] outputGradient) {
        int sequenceLength = cachedInputs.size();
        double[][] inputGradient = new double[sequenceLength][lstmInputSize];
        
        // Expand gradient if not returning sequences
        double[][] expandedGrad;
        if (!returnSequences) {
            expandedGrad = new double[sequenceLength][bidirectional ? hiddenSize * 2 : hiddenSize];
            System.arraycopy(outputGradient[0], 0, expandedGrad[sequenceLength - 1], 0, 
                           bidirectional ? hiddenSize * 2 : hiddenSize);
        } else {
            expandedGrad = outputGradient;
        }
        
        // Apply dropout gradient
        if (training && dropout > 0 && dropoutMask != null) {
            for (int t = 0; t < expandedGrad.length; t++) {
                for (int i = 0; i < expandedGrad[0].length; i++) {
                    expandedGrad[t][i] *= dropoutMask[t % dropoutMask.length][i];
                }
            }
        }
        
        // Split gradients for forward and backward cells
        double[][] forwardGrad = new double[sequenceLength][hiddenSize];
        double[][] backwardGrad = bidirectional ? new double[sequenceLength][hiddenSize] : null;
        
        for (int t = 0; t < sequenceLength; t++) {
            System.arraycopy(expandedGrad[t], 0, forwardGrad[t], 0, hiddenSize);
            if (bidirectional) {
                System.arraycopy(expandedGrad[t], hiddenSize, backwardGrad[t], 0, hiddenSize);
            }
        }
        
        // Backward through forward cell
        double[] dH = new double[hiddenSize];
        double[] dC = new double[hiddenSize];
        
        for (int t = sequenceLength - 1; t >= 0; t--) {
            // Add gradient from output
            for (int i = 0; i < hiddenSize; i++) {
                dH[i] += forwardGrad[t][i];
            }
            
            // Restore cell state for backward computation
            forwardCell.forward(cachedInputs.get(t), cachedHiddenStates.get(t), cachedCellStates.get(t));
            
            double[][] grads = forwardCell.backward(dH, dC);
            double[] dInput = grads[0];
            dH = grads[1];
            dC = grads[2];
            
            // Accumulate input gradient
            for (int i = 0; i < lstmInputSize; i++) {
                inputGradient[t][i] += dInput[i];
            }
        }
        
        // Backward through backward cell (for bidirectional)
        if (bidirectional) {
            dH = new double[hiddenSize];
            dC = new double[hiddenSize];
            
            for (int t = 0; t < sequenceLength; t++) {
                // Add gradient from output
                for (int i = 0; i < hiddenSize; i++) {
                    dH[i] += backwardGrad[t][i];
                }
                
                // Restore cell state for backward computation
                int backwardIdx = sequenceLength - 1 - t;
                backwardCell.forward(cachedInputs.get(backwardIdx), 
                                   backwardCachedHiddenStates.get(t + 1), 
                                   backwardCachedCellStates.get(t + 1));
                
                double[][] grads = backwardCell.backward(dH, dC);
                double[] dInput = grads[0];
                dH = grads[1];
                dC = grads[2];
                
                // Accumulate input gradient
                for (int i = 0; i < lstmInputSize; i++) {
                    inputGradient[backwardIdx][i] += dInput[i];
                }
            }
        }
        
        return inputGradient;
    }
    
    /**
     * Update weights using accumulated gradients.
     * 
     * @param batchSize Batch size for gradient averaging
     */
    public void updateWeights(int batchSize) {
        forwardCell.updateWeights(batchSize);
        if (bidirectional) {
            backwardCell.updateWeights(batchSize);
        }
    }
    
    /**
     * Reset accumulated gradients.
     */
    public void resetGradients() {
        forwardCell.resetGradients();
        if (bidirectional) {
            backwardCell.resetGradients();
        }
    }
    
    /**
     * Set training mode.
     */
    public void setTraining(boolean training) {
        this.training = training;
    }
    
    /**
     * Get the output size of this layer.
     */
    public int getOutputSize() {
        int outputDim = bidirectional ? hiddenSize * 2 : hiddenSize;
        return returnSequences ? -1 : outputDim;  // -1 indicates variable length
    }
    
    // ==================== Getters ====================
    
    public int getLSTMInputSize() {
        return lstmInputSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public boolean isReturnSequences() {
        return returnSequences;
    }
    
    public boolean isBidirectional() {
        return bidirectional;
    }
    
    public boolean isStateful() {
        return stateful;
    }
    
    public double getDropout() {
        return dropout;
    }
    
    public double[] getCurrentHiddenState() {
        return currentHiddenState.clone();
    }
    
    public double[] getCurrentCellState() {
        return currentCellState.clone();
    }
    
    public LSTMCell getForwardCell() {
        return forwardCell;
    }
    
    public LSTMCell getBackwardCell() {
        return backwardCell;
    }
    
    /**
     * Set learning rate for all cells.
     */
    public void setLearningRate(double learningRate) {
        forwardCell.setLearningRate(learningRate);
        if (bidirectional) {
            backwardCell.setLearningRate(learningRate);
        }
    }
    
    /**
     * Set gradient clipping value for all cells.
     */
    public void setClipValue(double clipValue) {
        forwardCell.setClipValue(clipValue);
        if (bidirectional) {
            backwardCell.setClipValue(clipValue);
        }
    }
}
