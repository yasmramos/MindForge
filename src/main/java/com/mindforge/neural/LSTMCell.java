package com.mindforge.neural;

import java.io.Serializable;
import java.util.Random;

/**
 * LSTM Cell implementation.
 * 
 * Long Short-Term Memory cell with forget, input, and output gates.
 * This is the fundamental building block of LSTM networks, designed to
 * capture long-term dependencies in sequential data.
 * 
 * Architecture:
 * - Forget Gate: Controls what information to discard from cell state
 * - Input Gate: Controls what new information to store in cell state
 * - Output Gate: Controls what information to output based on cell state
 * 
 * Equations:
 * f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  (forget gate)
 * i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  (input gate)
 * c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  (candidate cell state)
 * c_t = f_t * c_{t-1} + i_t * c̃_t  (new cell state)
 * o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  (output gate)
 * h_t = o_t * tanh(c_t)  (new hidden state)
 * 
 * @author MindForge
 * @version 1.0
 */
public class LSTMCell implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Dimensions
    private final int inputSize;
    private final int hiddenSize;
    
    // Weights for forget gate
    private double[][] Wf;  // [hiddenSize, inputSize + hiddenSize]
    private double[] bf;    // [hiddenSize]
    
    // Weights for input gate
    private double[][] Wi;
    private double[] bi;
    
    // Weights for candidate cell state
    private double[][] Wc;
    private double[] bc;
    
    // Weights for output gate
    private double[][] Wo;
    private double[] bo;
    
    // Gradients for weights
    private double[][] dWf, dWi, dWc, dWo;
    private double[] dbf, dbi, dbc, dbo;
    
    // Cache for backward pass
    private double[] lastInput;
    private double[] lastHiddenState;
    private double[] lastCellState;
    private double[] forgetGate;
    private double[] inputGate;
    private double[] candidateCell;
    private double[] outputGate;
    private double[] newCellState;
    private double[] newHiddenState;
    
    // Learning parameters
    private double learningRate = 0.001;
    private double clipValue = 5.0;  // Gradient clipping threshold
    
    /**
     * Creates a new LSTM cell.
     * 
     * @param inputSize Size of input vector
     * @param hiddenSize Size of hidden state vector
     */
    public LSTMCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        
        initializeWeights();
        initializeGradients();
    }
    
    /**
     * Initialize weights using Xavier/Glorot initialization.
     */
    private void initializeWeights() {
        Random random = new Random(42);
        int combinedSize = inputSize + hiddenSize;
        double scale = Math.sqrt(2.0 / (combinedSize + hiddenSize));
        
        // Initialize forget gate weights
        Wf = new double[hiddenSize][combinedSize];
        bf = new double[hiddenSize];
        initializeMatrix(Wf, random, scale);
        // Initialize forget gate bias to 1.0 for better gradient flow
        for (int i = 0; i < hiddenSize; i++) {
            bf[i] = 1.0;
        }
        
        // Initialize input gate weights
        Wi = new double[hiddenSize][combinedSize];
        bi = new double[hiddenSize];
        initializeMatrix(Wi, random, scale);
        
        // Initialize candidate cell state weights
        Wc = new double[hiddenSize][combinedSize];
        bc = new double[hiddenSize];
        initializeMatrix(Wc, random, scale);
        
        // Initialize output gate weights
        Wo = new double[hiddenSize][combinedSize];
        bo = new double[hiddenSize];
        initializeMatrix(Wo, random, scale);
    }
    
    /**
     * Initialize gradient matrices.
     */
    private void initializeGradients() {
        int combinedSize = inputSize + hiddenSize;
        
        dWf = new double[hiddenSize][combinedSize];
        dWi = new double[hiddenSize][combinedSize];
        dWc = new double[hiddenSize][combinedSize];
        dWo = new double[hiddenSize][combinedSize];
        
        dbf = new double[hiddenSize];
        dbi = new double[hiddenSize];
        dbc = new double[hiddenSize];
        dbo = new double[hiddenSize];
    }
    
    /**
     * Initialize a matrix with random values scaled by the given factor.
     */
    private void initializeMatrix(double[][] matrix, Random random, double scale) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = random.nextGaussian() * scale;
            }
        }
    }
    
    /**
     * Forward pass through the LSTM cell.
     * 
     * @param input Input vector at current time step
     * @param prevHiddenState Previous hidden state (h_{t-1})
     * @param prevCellState Previous cell state (c_{t-1})
     * @return Array containing [new hidden state, new cell state]
     */
    public double[][] forward(double[] input, double[] prevHiddenState, double[] prevCellState) {
        // Store for backward pass
        this.lastInput = input.clone();
        this.lastHiddenState = prevHiddenState.clone();
        this.lastCellState = prevCellState.clone();
        
        // Concatenate input and previous hidden state
        double[] combined = concatenate(input, prevHiddenState);
        
        // Compute gates
        forgetGate = sigmoid(matVecMul(Wf, combined, bf));
        inputGate = sigmoid(matVecMul(Wi, combined, bi));
        candidateCell = tanh(matVecMul(Wc, combined, bc));
        outputGate = sigmoid(matVecMul(Wo, combined, bo));
        
        // Compute new cell state: c_t = f_t * c_{t-1} + i_t * c̃_t
        newCellState = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            newCellState[i] = forgetGate[i] * prevCellState[i] + inputGate[i] * candidateCell[i];
        }
        
        // Compute new hidden state: h_t = o_t * tanh(c_t)
        newHiddenState = new double[hiddenSize];
        double[] tanhCell = tanh(newCellState);
        for (int i = 0; i < hiddenSize; i++) {
            newHiddenState[i] = outputGate[i] * tanhCell[i];
        }
        
        return new double[][] { newHiddenState, newCellState };
    }
    
    /**
     * Backward pass through the LSTM cell.
     * 
     * @param dHidden Gradient of loss with respect to hidden state
     * @param dCell Gradient of loss with respect to cell state (from next time step)
     * @return Array containing [gradient w.r.t. input, gradient w.r.t. prev hidden, gradient w.r.t. prev cell]
     */
    public double[][] backward(double[] dHidden, double[] dCell) {
        // Compute tanh of cell state
        double[] tanhCell = tanh(newCellState);
        
        // Gradient of output gate
        double[] dOutputGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dOutputGate[i] = dHidden[i] * tanhCell[i] * sigmoidDerivative(outputGate[i]);
        }
        
        // Gradient of cell state
        double[] dCellTotal = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dCellTotal[i] = dCell[i] + dHidden[i] * outputGate[i] * tanhDerivative(tanhCell[i]);
        }
        
        // Gradient of forget gate
        double[] dForgetGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dForgetGate[i] = dCellTotal[i] * lastCellState[i] * sigmoidDerivative(forgetGate[i]);
        }
        
        // Gradient of input gate
        double[] dInputGate = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dInputGate[i] = dCellTotal[i] * candidateCell[i] * sigmoidDerivative(inputGate[i]);
        }
        
        // Gradient of candidate cell state
        double[] dCandidateCell = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dCandidateCell[i] = dCellTotal[i] * inputGate[i] * tanhDerivative(candidateCell[i]);
        }
        
        // Concatenated input for weight gradients
        double[] combined = concatenate(lastInput, lastHiddenState);
        
        // Compute weight gradients
        accumulateGradients(dWf, dbf, dForgetGate, combined);
        accumulateGradients(dWi, dbi, dInputGate, combined);
        accumulateGradients(dWc, dbc, dCandidateCell, combined);
        accumulateGradients(dWo, dbo, dOutputGate, combined);
        
        // Compute gradient w.r.t. combined input
        double[] dCombined = new double[inputSize + hiddenSize];
        addMatVecMulTranspose(dCombined, Wf, dForgetGate);
        addMatVecMulTranspose(dCombined, Wi, dInputGate);
        addMatVecMulTranspose(dCombined, Wc, dCandidateCell);
        addMatVecMulTranspose(dCombined, Wo, dOutputGate);
        
        // Split gradient into input and previous hidden state
        double[] dInput = new double[inputSize];
        double[] dPrevHidden = new double[hiddenSize];
        System.arraycopy(dCombined, 0, dInput, 0, inputSize);
        System.arraycopy(dCombined, inputSize, dPrevHidden, 0, hiddenSize);
        
        // Gradient w.r.t. previous cell state
        double[] dPrevCell = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            dPrevCell[i] = dCellTotal[i] * forgetGate[i];
        }
        
        return new double[][] { dInput, dPrevHidden, dPrevCell };
    }
    
    /**
     * Update weights using accumulated gradients.
     * 
     * @param batchSize Size of the batch for gradient averaging
     */
    public void updateWeights(int batchSize) {
        double scale = learningRate / batchSize;
        
        // Update with gradient clipping
        updateMatrix(Wf, dWf, scale);
        updateVector(bf, dbf, scale);
        
        updateMatrix(Wi, dWi, scale);
        updateVector(bi, dbi, scale);
        
        updateMatrix(Wc, dWc, scale);
        updateVector(bc, dbc, scale);
        
        updateMatrix(Wo, dWo, scale);
        updateVector(bo, dbo, scale);
        
        // Reset gradients
        resetGradients();
    }
    
    /**
     * Reset all accumulated gradients to zero.
     */
    public void resetGradients() {
        zeroMatrix(dWf);
        zeroMatrix(dWi);
        zeroMatrix(dWc);
        zeroMatrix(dWo);
        
        zeroVector(dbf);
        zeroVector(dbi);
        zeroVector(dbc);
        zeroVector(dbo);
    }
    
    /**
     * Get initial hidden state (zeros).
     */
    public double[] getInitialHiddenState() {
        return new double[hiddenSize];
    }
    
    /**
     * Get initial cell state (zeros).
     */
    public double[] getInitialCellState() {
        return new double[hiddenSize];
    }
    
    // ==================== Helper Methods ====================
    
    private double[] concatenate(double[] a, double[] b) {
        double[] result = new double[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
    
    private double[] matVecMul(double[][] matrix, double[] vector, double[] bias) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = bias[i];
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }
    
    private void addMatVecMulTranspose(double[] result, double[][] matrix, double[] vector) {
        for (int j = 0; j < result.length; j++) {
            for (int i = 0; i < vector.length; i++) {
                result[j] += matrix[i][j] * vector[i];
            }
        }
    }
    
    private void accumulateGradients(double[][] dW, double[] db, double[] delta, double[] input) {
        for (int i = 0; i < delta.length; i++) {
            db[i] += delta[i];
            for (int j = 0; j < input.length; j++) {
                dW[i][j] += delta[i] * input[j];
            }
        }
    }
    
    private void updateMatrix(double[][] W, double[][] dW, double scale) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                double grad = clip(dW[i][j]);
                W[i][j] -= scale * grad;
            }
        }
    }
    
    private void updateVector(double[] b, double[] db, double scale) {
        for (int i = 0; i < b.length; i++) {
            double grad = clip(db[i]);
            b[i] -= scale * grad;
        }
    }
    
    private double clip(double value) {
        return Math.max(-clipValue, Math.min(clipValue, value));
    }
    
    private void zeroMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                matrix[i][j] = 0;
            }
        }
    }
    
    private void zeroVector(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = 0;
        }
    }
    
    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
        }
        return result;
    }
    
    private double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1 - sigmoidOutput);
    }
    
    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
        }
        return result;
    }
    
    private double tanhDerivative(double tanhOutput) {
        return 1 - tanhOutput * tanhOutput;
    }
    
    // ==================== Getters and Setters ====================
    
    public int getInputSize() {
        return inputSize;
    }
    
    public int getHiddenSize() {
        return hiddenSize;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public double getClipValue() {
        return clipValue;
    }
    
    public void setClipValue(double clipValue) {
        this.clipValue = clipValue;
    }
    
    /**
     * Get weights for serialization.
     */
    public double[][][] getWeights() {
        return new double[][][] { Wf, Wi, Wc, Wo };
    }
    
    /**
     * Get biases for serialization.
     */
    public double[][] getBiases() {
        return new double[][] { bf, bi, bc, bo };
    }
    
    /**
     * Set weights from serialization.
     */
    public void setWeights(double[][][] weights) {
        this.Wf = weights[0];
        this.Wi = weights[1];
        this.Wc = weights[2];
        this.Wo = weights[3];
    }
    
    /**
     * Set biases from serialization.
     */
    public void setBiases(double[][] biases) {
        this.bf = biases[0];
        this.bi = biases[1];
        this.bc = biases[2];
        this.bo = biases[3];
    }
}
