package com.mindforge.neural;

import java.io.*;
import java.util.*;

/**
 * Recurrent Neural Network implementation with LSTM support.
 * 
 * This class provides a flexible RNN architecture with LSTM layers,
 * supporting various configurations for sequence modeling tasks:
 * - Sequence classification
 * - Sequence-to-sequence modeling
 * - Time series prediction
 * - Text processing
 * 
 * Features:
 * - Builder pattern for easy configuration
 * - Multiple LSTM layers (stacking)
 * - Dense output layers
 * - Bidirectional processing
 * - Dropout regularization
 * - Gradient clipping
 * - Serialization support
 * 
 * @author MindForge
 * @version 1.0
 */
public class RNN implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Network architecture
    private List<LSTMLayer> lstmLayers;
    private List<DenseLayer> denseLayers;
    private int inputSize;
    private int sequenceLength;
    private int outputSize;
    
    // Training configuration
    private double learningRate = 0.001;
    private double clipValue = 5.0;
    private int batchSize = 32;
    private int epochs = 100;
    private boolean verbose = true;
    
    // Training state
    private boolean training = true;
    private double[] trainingLosses;
    
    /**
     * Private constructor - use Builder pattern.
     */
    private RNN() {
        this.lstmLayers = new ArrayList<>();
        this.denseLayers = new ArrayList<>();
    }
    
    /**
     * Forward pass through the entire network.
     * 
     * @param sequence Input sequence [sequenceLength, inputSize]
     * @return Output vector
     */
    public double[] forward(double[][] sequence) {
        // Forward through LSTM layers
        double[][] currentOutput = sequence;
        
        for (int i = 0; i < lstmLayers.size(); i++) {
            LSTMLayer lstm = lstmLayers.get(i);
            lstm.setTraining(training);
            currentOutput = lstm.forwardSequence(currentOutput);
        }
        
        // Get the final output (last timestep or full sequence depending on last LSTM config)
        double[] flatOutput;
        if (currentOutput.length == 1) {
            flatOutput = currentOutput[0];
        } else {
            // Use last timestep output
            flatOutput = currentOutput[currentOutput.length - 1];
        }
        
        // Forward through dense layers
        for (DenseLayer dense : denseLayers) {
            flatOutput = dense.forward(flatOutput);
        }
        
        return flatOutput;
    }
    
    /**
     * Forward pass for 1D input (flattened sequence).
     * 
     * @param input Flattened input [sequenceLength * inputSize]
     * @return Output vector
     */
    public double[] forward(double[] input) {
        // Reshape to 2D
        double[][] sequence = new double[sequenceLength][inputSize];
        for (int t = 0; t < sequenceLength; t++) {
            System.arraycopy(input, t * inputSize, sequence[t], 0, inputSize);
        }
        return forward(sequence);
    }
    
    /**
     * Backward pass through the entire network.
     * 
     * @param target Target output
     * @param prediction Network prediction
     * @return Loss value
     */
    private double backward(double[] target, double[] prediction) {
        // Compute loss and initial gradient (cross-entropy for classification)
        double loss = 0;
        double[] gradient = new double[prediction.length];
        
        // Softmax cross-entropy gradient
        for (int i = 0; i < prediction.length; i++) {
            gradient[i] = prediction[i] - target[i];
            if (target[i] > 0) {
                loss -= target[i] * Math.log(Math.max(prediction[i], 1e-15));
            }
        }
        
        // Backward through dense layers (DenseLayer.backward updates weights internally)
        for (int i = denseLayers.size() - 1; i >= 0; i--) {
            gradient = denseLayers.get(i).backward(gradient, learningRate);
        }
        
        // Backward through LSTM layers
        LSTMLayer lastLstm = lstmLayers.get(lstmLayers.size() - 1);
        double[][] seqGradient;
        
        if (lastLstm.isReturnSequences()) {
            // For sequence-to-sequence, gradient applies to all timesteps
            // This is a simplified version - full seq2seq would need more handling
            seqGradient = new double[sequenceLength][gradient.length];
            seqGradient[sequenceLength - 1] = gradient;
        } else {
            seqGradient = new double[1][gradient.length];
            seqGradient[0] = gradient;
        }
        
        for (int i = lstmLayers.size() - 1; i >= 0; i--) {
            seqGradient = lstmLayers.get(i).backwardSequence(seqGradient);
        }
        
        return loss;
    }
    
    /**
     * Train the network on a dataset.
     * 
     * @param sequences Training sequences [numSamples][sequenceLength][inputSize]
     * @param labels Training labels (one-hot encoded) [numSamples][numClasses]
     */
    public void train(double[][][] sequences, double[][] labels) {
        training = true;
        int numSamples = sequences.length;
        trainingLosses = new double[epochs];
        
        // Set learning rates
        for (LSTMLayer lstm : lstmLayers) {
            lstm.setLearningRate(learningRate);
            lstm.setClipValue(clipValue);
        }
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;
            int numBatches = (numSamples + batchSize - 1) / batchSize;
            
            // Shuffle indices
            Integer[] indices = new Integer[numSamples];
            for (int i = 0; i < numSamples; i++) indices[i] = i;
            Collections.shuffle(Arrays.asList(indices));
            
            for (int batch = 0; batch < numBatches; batch++) {
                int start = batch * batchSize;
                int end = Math.min(start + batchSize, numSamples);
                int currentBatchSize = end - start;
                
                double batchLoss = 0;
                
                // Reset LSTM gradients
                for (LSTMLayer lstm : lstmLayers) {
                    lstm.resetGradients();
                }
                
                // Process batch
                for (int i = start; i < end; i++) {
                    int idx = indices[i];
                    
                    // Reset states for each sample
                    for (LSTMLayer lstm : lstmLayers) {
                        lstm.resetStates();
                    }
                    
                    // Forward pass
                    double[] output = forward(sequences[idx]);
                    
                    // Apply softmax if needed
                    double[] prediction = softmax(output);
                    
                    // Backward pass
                    batchLoss += backward(labels[idx], prediction);
                }
                
                // Update LSTM weights (DenseLayer weights are updated in backward())
                for (LSTMLayer lstm : lstmLayers) {
                    lstm.updateWeights(currentBatchSize);
                }
                
                epochLoss += batchLoss;
            }
            
            epochLoss /= numSamples;
            trainingLosses[epoch] = epochLoss;
            
            if (verbose && (epoch + 1) % 10 == 0) {
                System.out.printf("Epoch %d/%d - Loss: %.6f%n", epoch + 1, epochs, epochLoss);
            }
        }
    }
    
    /**
     * Train with 1D flattened sequences.
     * 
     * @param sequences Flattened sequences [numSamples][sequenceLength * inputSize]
     * @param labels Training labels
     */
    public void train(double[][] sequences, double[][] labels) {
        // Reshape to 3D
        int numSamples = sequences.length;
        double[][][] sequences3D = new double[numSamples][sequenceLength][inputSize];
        
        for (int s = 0; s < numSamples; s++) {
            for (int t = 0; t < sequenceLength; t++) {
                System.arraycopy(sequences[s], t * inputSize, sequences3D[s][t], 0, inputSize);
            }
        }
        
        train(sequences3D, labels);
    }
    
    /**
     * Predict class probabilities for a sequence.
     * 
     * @param sequence Input sequence
     * @return Class probabilities
     */
    public double[] predict(double[][] sequence) {
        training = false;
        
        // Reset states
        for (LSTMLayer lstm : lstmLayers) {
            lstm.resetStates();
        }
        
        double[] output = forward(sequence);
        return softmax(output);
    }
    
    /**
     * Predict class for a sequence.
     * 
     * @param sequence Input sequence
     * @return Predicted class index
     */
    public int predictClass(double[][] sequence) {
        double[] probs = predict(sequence);
        int maxIdx = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Predict classes for multiple sequences.
     * 
     * @param sequences Input sequences
     * @return Predicted class indices
     */
    public int[] predictClasses(double[][][] sequences) {
        int[] predictions = new int[sequences.length];
        for (int i = 0; i < sequences.length; i++) {
            predictions[i] = predictClass(sequences[i]);
        }
        return predictions;
    }
    
    /**
     * Evaluate accuracy on a test set.
     * 
     * @param sequences Test sequences
     * @param labels Test labels (one-hot)
     * @return Accuracy (0.0 to 1.0)
     */
    public double evaluate(double[][][] sequences, double[][] labels) {
        int correct = 0;
        for (int i = 0; i < sequences.length; i++) {
            int predicted = predictClass(sequences[i]);
            int actual = argmax(labels[i]);
            if (predicted == actual) {
                correct++;
            }
        }
        return (double) correct / sequences.length;
    }
    
    /**
     * Softmax activation function.
     */
    private double[] softmax(double[] x) {
        double[] result = new double[x.length];
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) max = Math.max(max, v);
        
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i] - max);
            sum += result[i];
        }
        
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    /**
     * Find index of maximum value.
     */
    private int argmax(double[] arr) {
        int maxIdx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Save the model to a file.
     * 
     * @param filename File path
     */
    public void save(String filename) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(this);
        }
    }
    
    /**
     * Load a model from a file.
     * 
     * @param filename File path
     * @return Loaded RNN model
     */
    public static RNN load(String filename) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            return (RNN) ois.readObject();
        }
    }
    
    // ==================== Getters ====================
    
    public int getInputSize() {
        return inputSize;
    }
    
    public int getSequenceLength() {
        return sequenceLength;
    }
    
    public int getOutputSize() {
        return outputSize;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public int getEpochs() {
        return epochs;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public double[] getTrainingLosses() {
        return trainingLosses;
    }
    
    public int getNumLSTMLayers() {
        return lstmLayers.size();
    }
    
    public int getNumDenseLayers() {
        return denseLayers.size();
    }
    
    public List<LSTMLayer> getLSTMLayers() {
        return new ArrayList<>(lstmLayers);
    }
    
    /**
     * Get summary of the network architecture.
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=".repeat(60)).append("\n");
        sb.append("RNN Model Summary\n");
        sb.append("=".repeat(60)).append("\n");
        sb.append(String.format("Input Shape: [%d, %d]\n", sequenceLength, inputSize));
        sb.append("-".repeat(60)).append("\n");
        
        int layerNum = 1;
        for (LSTMLayer lstm : lstmLayers) {
            String type = lstm.isBidirectional() ? "Bidirectional LSTM" : "LSTM";
            int outputDim = lstm.isBidirectional() ? lstm.getHiddenSize() * 2 : lstm.getHiddenSize();
            String returnSeq = lstm.isReturnSequences() ? ", return_sequences=true" : "";
            String dropout = lstm.getDropout() > 0 ? String.format(", dropout=%.2f", lstm.getDropout()) : "";
            
            sb.append(String.format("Layer %d: %s (hidden=%d%s%s)\n", 
                layerNum++, type, lstm.getHiddenSize(), returnSeq, dropout));
            sb.append(String.format("         Output: [%s, %d]\n",
                lstm.isReturnSequences() ? String.valueOf(sequenceLength) : "1", outputDim));
        }
        
        for (DenseLayer dense : denseLayers) {
            sb.append(String.format("Layer %d: Dense (units=%d, activation=%s)\n",
                layerNum++, dense.getOutputSize(), dense.getActivation()));
        }
        
        sb.append("-".repeat(60)).append("\n");
        sb.append(String.format("Output Shape: [%d]\n", outputSize));
        sb.append("=".repeat(60)).append("\n");
        
        return sb.toString();
    }
    
    // ==================== Builder Pattern ====================
    
    /**
     * Create a new RNN builder.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Builder class for RNN configuration.
     */
    public static class Builder {
        private RNN rnn;
        private int currentInputSize;
        private int currentSequenceLength;
        private boolean lastReturnSequences = true;
        
        private Builder() {
            this.rnn = new RNN();
        }
        
        /**
         * Set input shape.
         * 
         * @param sequenceLength Length of input sequences
         * @param inputSize Size of input features
         * @return Builder instance
         */
        public Builder inputShape(int sequenceLength, int inputSize) {
            this.rnn.sequenceLength = sequenceLength;
            this.rnn.inputSize = inputSize;
            this.currentSequenceLength = sequenceLength;
            this.currentInputSize = inputSize;
            return this;
        }
        
        /**
         * Add an LSTM layer.
         * 
         * @param hiddenSize Size of hidden state
         * @return Builder instance
         */
        public Builder addLSTM(int hiddenSize) {
            return addLSTM(hiddenSize, false, false, false, 0.0);
        }
        
        /**
         * Add an LSTM layer with return sequences option.
         * 
         * @param hiddenSize Size of hidden state
         * @param returnSequences Whether to return full sequence
         * @return Builder instance
         */
        public Builder addLSTM(int hiddenSize, boolean returnSequences) {
            return addLSTM(hiddenSize, returnSequences, false, false, 0.0);
        }
        
        /**
         * Add an LSTM layer with full configuration.
         * 
         * @param hiddenSize Size of hidden state
         * @param returnSequences Whether to return full sequence
         * @param bidirectional Whether to use bidirectional processing
         * @param stateful Whether to maintain state between batches
         * @param dropout Dropout rate
         * @return Builder instance
         */
        public Builder addLSTM(int hiddenSize, boolean returnSequences, 
                               boolean bidirectional, boolean stateful, double dropout) {
            if (rnn.inputSize == 0) {
                throw new IllegalStateException("Must call inputShape() before adding layers");
            }
            
            if (!rnn.lstmLayers.isEmpty() && !lastReturnSequences) {
                throw new IllegalStateException("Cannot add LSTM after a non-return-sequences LSTM layer");
            }
            
            LSTMLayer lstm = new LSTMLayer(currentInputSize, hiddenSize, 
                                          returnSequences, bidirectional, stateful, dropout);
            rnn.lstmLayers.add(lstm);
            
            this.currentInputSize = bidirectional ? hiddenSize * 2 : hiddenSize;
            this.lastReturnSequences = returnSequences;
            
            return this;
        }
        
        /**
         * Add a bidirectional LSTM layer.
         * 
         * @param hiddenSize Size of hidden state (per direction)
         * @return Builder instance
         */
        public Builder addBidirectionalLSTM(int hiddenSize) {
            return addLSTM(hiddenSize, true, true, false, 0.0);
        }
        
        /**
         * Add a bidirectional LSTM layer with full configuration.
         * 
         * @param hiddenSize Size of hidden state (per direction)
         * @param returnSequences Whether to return full sequence
         * @param dropout Dropout rate
         * @return Builder instance
         */
        public Builder addBidirectionalLSTM(int hiddenSize, boolean returnSequences, double dropout) {
            return addLSTM(hiddenSize, returnSequences, true, false, dropout);
        }
        
        /**
         * Add a dense (fully connected) layer.
         * 
         * @param units Number of units
         * @return Builder instance
         */
        public Builder addDense(int units) {
            return addDense(units, ActivationFunction.RELU);
        }
        
        /**
         * Add a dense layer with specified activation.
         * 
         * @param units Number of units
         * @param activation Activation function
         * @return Builder instance
         */
        public Builder addDense(int units, ActivationFunction activation) {
            if (rnn.lstmLayers.isEmpty()) {
                throw new IllegalStateException("Must add at least one LSTM layer before dense layers");
            }
            
            DenseLayer dense = new DenseLayer(currentInputSize, units, activation);
            rnn.denseLayers.add(dense);
            this.currentInputSize = units;
            
            return this;
        }
        
        /**
         * Add output layer (dense with softmax).
         * 
         * @param numClasses Number of output classes
         * @return Builder instance
         */
        public Builder addOutput(int numClasses) {
            rnn.outputSize = numClasses;
            // Add a linear output layer - softmax is applied in forward/predict
            return addDense(numClasses, ActivationFunction.LINEAR);
        }
        
        /**
         * Set learning rate.
         * 
         * @param learningRate Learning rate
         * @return Builder instance
         */
        public Builder learningRate(double learningRate) {
            rnn.learningRate = learningRate;
            return this;
        }
        
        /**
         * Set gradient clipping value.
         * 
         * @param clipValue Gradient clipping threshold
         * @return Builder instance
         */
        public Builder clipValue(double clipValue) {
            rnn.clipValue = clipValue;
            return this;
        }
        
        /**
         * Set batch size.
         * 
         * @param batchSize Batch size
         * @return Builder instance
         */
        public Builder batchSize(int batchSize) {
            rnn.batchSize = batchSize;
            return this;
        }
        
        /**
         * Set number of training epochs.
         * 
         * @param epochs Number of epochs
         * @return Builder instance
         */
        public Builder epochs(int epochs) {
            rnn.epochs = epochs;
            return this;
        }
        
        /**
         * Set verbose mode for training output.
         * 
         * @param verbose Whether to print training progress
         * @return Builder instance
         */
        public Builder verbose(boolean verbose) {
            rnn.verbose = verbose;
            return this;
        }
        
        /**
         * Build the RNN model.
         * 
         * @return Configured RNN instance
         */
        public RNN build() {
            if (rnn.lstmLayers.isEmpty()) {
                throw new IllegalStateException("Must add at least one LSTM layer");
            }
            if (rnn.outputSize == 0) {
                throw new IllegalStateException("Must add an output layer");
            }
            
            return rnn;
        }
    }
    
    // ==================== Utility Methods for Sequence Data ====================
    
    /**
     * Create one-hot encoding for labels.
     * 
     * @param labels Integer labels
     * @param numClasses Number of classes
     * @return One-hot encoded labels
     */
    public static double[][] oneHotEncode(int[] labels, int numClasses) {
        double[][] encoded = new double[labels.length][numClasses];
        for (int i = 0; i < labels.length; i++) {
            encoded[i][labels[i]] = 1.0;
        }
        return encoded;
    }
    
    /**
     * Pad sequences to the same length.
     * 
     * @param sequences Variable length sequences
     * @param maxLength Maximum sequence length (0 for auto)
     * @param padValue Value to use for padding
     * @return Padded sequences
     */
    public static double[][][] padSequences(double[][][] sequences, int maxLength, double padValue) {
        if (maxLength <= 0) {
            maxLength = 0;
            for (double[][] seq : sequences) {
                maxLength = Math.max(maxLength, seq.length);
            }
        }
        
        int inputSize = sequences[0][0].length;
        double[][][] padded = new double[sequences.length][maxLength][inputSize];
        
        for (int i = 0; i < sequences.length; i++) {
            int seqLen = Math.min(sequences[i].length, maxLength);
            int startIdx = maxLength - seqLen;  // Right padding
            
            // Fill with pad value
            for (int t = 0; t < startIdx; t++) {
                Arrays.fill(padded[i][t], padValue);
            }
            
            // Copy sequence
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(sequences[i][t], 0, padded[i][startIdx + t], 0, inputSize);
            }
        }
        
        return padded;
    }
    
    /**
     * Create sliding window sequences from time series data.
     * 
     * @param data Time series data [totalLength, features]
     * @param windowSize Size of sliding window
     * @param step Step size between windows
     * @return Windowed sequences [numWindows, windowSize, features]
     */
    public static double[][][] createSlidingWindows(double[][] data, int windowSize, int step) {
        int numWindows = (data.length - windowSize) / step + 1;
        int features = data[0].length;
        
        double[][][] windows = new double[numWindows][windowSize][features];
        
        for (int w = 0; w < numWindows; w++) {
            int start = w * step;
            for (int t = 0; t < windowSize; t++) {
                System.arraycopy(data[start + t], 0, windows[w][t], 0, features);
            }
        }
        
        return windows;
    }
}
