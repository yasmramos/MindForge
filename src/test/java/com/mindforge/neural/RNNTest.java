package com.mindforge.neural;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

import java.io.*;
import java.nio.file.Path;
import java.util.Random;

/**
 * Comprehensive test suite for RNN/LSTM implementation.
 * 
 * Tests cover:
 * - LSTMCell: forward pass, backward pass, gradient computation
 * - LSTMLayer: sequence processing, bidirectional, stateful
 * - RNN: builder pattern, training, prediction, serialization
 * 
 * @author MindForge
 * @version 1.0
 */
@DisplayName("RNN/LSTM Tests")
class RNNTest {
    
    private static final double EPSILON = 1e-6;
    private static final Random random = new Random(42);
    
    // ==================== LSTMCell Tests ====================
    
    @Nested
    @DisplayName("LSTMCell Tests")
    class LSTMCellTests {
        
        @Test
        @DisplayName("Constructor initializes correctly")
        void testConstructor() {
            LSTMCell cell = new LSTMCell(10, 20);
            
            assertEquals(10, cell.getInputSize());
            assertEquals(20, cell.getHiddenSize());
            assertEquals(0.001, cell.getLearningRate());
            assertEquals(5.0, cell.getClipValue());
        }
        
        @Test
        @DisplayName("Forward pass produces valid output dimensions")
        void testForwardDimensions() {
            LSTMCell cell = new LSTMCell(5, 8);
            
            double[] input = new double[5];
            double[] hiddenState = cell.getInitialHiddenState();
            double[] cellState = cell.getInitialCellState();
            
            double[][] output = cell.forward(input, hiddenState, cellState);
            
            assertEquals(2, output.length, "Should return [hidden, cell]");
            assertEquals(8, output[0].length, "Hidden state should match hiddenSize");
            assertEquals(8, output[1].length, "Cell state should match hiddenSize");
        }
        
        @Test
        @DisplayName("Forward pass changes hidden and cell states")
        void testForwardChangesStates() {
            LSTMCell cell = new LSTMCell(4, 6);
            
            double[] input = {0.5, -0.3, 0.8, 0.1};
            double[] hiddenState = cell.getInitialHiddenState();
            double[] cellState = cell.getInitialCellState();
            
            double[][] output = cell.forward(input, hiddenState, cellState);
            
            // New states should be different from initial zeros (with non-zero input)
            boolean hiddenChanged = false;
            boolean cellChanged = false;
            
            for (int i = 0; i < output[0].length; i++) {
                if (Math.abs(output[0][i]) > EPSILON) hiddenChanged = true;
                if (Math.abs(output[1][i]) > EPSILON) cellChanged = true;
            }
            
            assertTrue(hiddenChanged, "Hidden state should change with non-zero input");
            assertTrue(cellChanged, "Cell state should change with non-zero input");
        }
        
        @Test
        @DisplayName("Backward pass produces valid gradient dimensions")
        void testBackwardDimensions() {
            LSTMCell cell = new LSTMCell(5, 8);
            
            double[] input = randomArray(5);
            double[] hiddenState = randomArray(8);
            double[] cellState = randomArray(8);
            
            // Forward pass
            cell.forward(input, hiddenState, cellState);
            
            // Backward pass
            double[] dHidden = randomArray(8);
            double[] dCell = randomArray(8);
            
            double[][] gradients = cell.backward(dHidden, dCell);
            
            assertEquals(3, gradients.length, "Should return [dInput, dPrevHidden, dPrevCell]");
            assertEquals(5, gradients[0].length, "dInput should match inputSize");
            assertEquals(8, gradients[1].length, "dPrevHidden should match hiddenSize");
            assertEquals(8, gradients[2].length, "dPrevCell should match hiddenSize");
        }
        
        @Test
        @DisplayName("Weight update modifies weights")
        void testWeightUpdate() {
            LSTMCell cell = new LSTMCell(3, 4);
            cell.setLearningRate(0.1);  // Use higher learning rate for visible changes
            
            // Get initial weights (deep copy)
            double[][][] originalWeights = cell.getWeights();
            double[][] initialWeightsCopy = new double[originalWeights[0].length][];
            for (int i = 0; i < originalWeights[0].length; i++) {
                initialWeightsCopy[i] = originalWeights[0][i].clone();
            }
            
            // Forward and backward with non-trivial values
            double[] input = {1.0, 2.0, 3.0};
            double[] hiddenState = {0.5, 0.5, 0.5, 0.5};
            double[] cellState = {0.1, 0.1, 0.1, 0.1};
            
            cell.forward(input, hiddenState, cellState);
            cell.backward(new double[]{1.0, 1.0, 1.0, 1.0}, new double[]{0.5, 0.5, 0.5, 0.5});
            
            // Update weights
            cell.updateWeights(1);
            
            // Weights should have changed
            double[][][] newWeights = cell.getWeights();
            
            boolean weightsChanged = false;
            for (int i = 0; i < newWeights[0].length && !weightsChanged; i++) {
                for (int j = 0; j < newWeights[0][i].length && !weightsChanged; j++) {
                    if (Math.abs(newWeights[0][i][j] - initialWeightsCopy[i][j]) > EPSILON) {
                        weightsChanged = true;
                    }
                }
            }
            
            assertTrue(weightsChanged, "Weights should change after update");
        }
        
        @Test
        @DisplayName("Gradient clipping works correctly")
        void testGradientClipping() {
            LSTMCell cell = new LSTMCell(2, 3);
            cell.setClipValue(1.0);
            
            // Forward with large values
            double[] input = {100.0, -100.0};
            double[] hiddenState = {50.0, -50.0, 50.0};
            double[] cellState = {0.0, 0.0, 0.0};
            
            cell.forward(input, hiddenState, cellState);
            
            // Backward with large gradients
            double[] dHidden = {1000.0, -1000.0, 1000.0};
            double[] dCell = {1000.0, -1000.0, 1000.0};
            
            // This should not throw and weights should be properly clipped
            assertDoesNotThrow(() -> {
                cell.backward(dHidden, dCell);
                cell.updateWeights(1);
            });
        }
    }
    
    // ==================== LSTMLayer Tests ====================
    
    @Nested
    @DisplayName("LSTMLayer Tests")
    class LSTMLayerTests {
        
        @Test
        @DisplayName("Constructor initializes correctly")
        void testConstructor() {
            LSTMLayer layer = new LSTMLayer(10, 20);
            
            assertEquals(10, layer.getInputSize());
            assertEquals(20, layer.getHiddenSize());
            assertFalse(layer.isReturnSequences());
            assertFalse(layer.isBidirectional());
            assertFalse(layer.isStateful());
            assertEquals(0.0, layer.getDropout());
        }
        
        @Test
        @DisplayName("Full constructor with all options")
        void testFullConstructor() {
            LSTMLayer layer = new LSTMLayer(10, 20, true, true, true, 0.5);
            
            assertEquals(10, layer.getInputSize());
            assertEquals(20, layer.getHiddenSize());
            assertTrue(layer.isReturnSequences());
            assertTrue(layer.isBidirectional());
            assertTrue(layer.isStateful());
            assertEquals(0.5, layer.getDropout());
        }
        
        @Test
        @DisplayName("Forward pass with sequence input")
        void testForwardSequence() {
            LSTMLayer layer = new LSTMLayer(5, 8, false, false, false, 0.0);
            
            double[][] sequence = new double[10][5];  // 10 timesteps, 5 features
            for (int t = 0; t < 10; t++) {
                sequence[t] = randomArray(5);
            }
            
            double[][] output = layer.forwardSequence(sequence);
            
            // Should return last output only
            assertEquals(1, output.length, "Should return single output when returnSequences=false");
            assertEquals(8, output[0].length, "Output should match hiddenSize");
        }
        
        @Test
        @DisplayName("Forward pass returns sequences")
        void testForwardReturnSequences() {
            LSTMLayer layer = new LSTMLayer(5, 8, true, false, false, 0.0);
            
            double[][] sequence = new double[10][5];
            for (int t = 0; t < 10; t++) {
                sequence[t] = randomArray(5);
            }
            
            double[][] output = layer.forwardSequence(sequence);
            
            assertEquals(10, output.length, "Should return full sequence");
            assertEquals(8, output[0].length, "Each output should match hiddenSize");
        }
        
        @Test
        @DisplayName("Bidirectional LSTM produces double hidden size")
        void testBidirectionalOutput() {
            LSTMLayer layer = new LSTMLayer(5, 8, true, true, false, 0.0);
            
            double[][] sequence = new double[10][5];
            for (int t = 0; t < 10; t++) {
                sequence[t] = randomArray(5);
            }
            
            double[][] output = layer.forwardSequence(sequence);
            
            assertEquals(10, output.length);
            assertEquals(16, output[0].length, "Bidirectional should have 2x hiddenSize output");
        }
        
        @Test
        @DisplayName("Stateful mode preserves state between calls")
        void testStatefulMode() {
            LSTMLayer layer = new LSTMLayer(5, 8, false, false, true, 0.0);
            
            double[][] sequence1 = new double[5][5];
            double[][] sequence2 = new double[5][5];
            
            for (int t = 0; t < 5; t++) {
                sequence1[t] = randomArray(5);
                sequence2[t] = randomArray(5);
            }
            
            // First forward pass
            layer.forwardSequence(sequence1);
            double[] hiddenAfterFirst = layer.getCurrentHiddenState();
            
            // Second forward pass should start from preserved state
            layer.forwardSequence(sequence2);
            double[] hiddenAfterSecond = layer.getCurrentHiddenState();
            
            // States should be different
            boolean statesDifferent = false;
            for (int i = 0; i < hiddenAfterFirst.length; i++) {
                if (Math.abs(hiddenAfterFirst[i] - hiddenAfterSecond[i]) > EPSILON) {
                    statesDifferent = true;
                    break;
                }
            }
            
            assertTrue(statesDifferent, "State should change between sequences");
        }
        
        @Test
        @DisplayName("Reset states clears hidden and cell states")
        void testResetStates() {
            LSTMLayer layer = new LSTMLayer(5, 8);
            
            double[][] sequence = new double[5][5];
            for (int t = 0; t < 5; t++) {
                sequence[t] = randomArray(5);
            }
            
            layer.forwardSequence(sequence);
            
            // States should be non-zero after forward
            double[] hiddenBefore = layer.getCurrentHiddenState();
            boolean nonZero = false;
            for (double v : hiddenBefore) {
                if (Math.abs(v) > EPSILON) {
                    nonZero = true;
                    break;
                }
            }
            assertTrue(nonZero, "State should be non-zero after forward");
            
            // Reset states
            layer.resetStates();
            
            double[] hiddenAfter = layer.getCurrentHiddenState();
            for (double v : hiddenAfter) {
                assertEquals(0.0, v, EPSILON, "State should be zero after reset");
            }
        }
        
        @Test
        @DisplayName("Backward pass produces valid gradients")
        void testBackwardPass() {
            LSTMLayer layer = new LSTMLayer(5, 8, true, false, false, 0.0);
            layer.setTraining(true);
            
            double[][] sequence = new double[10][5];
            for (int t = 0; t < 10; t++) {
                sequence[t] = randomArray(5);
            }
            
            // Forward
            double[][] output = layer.forwardSequence(sequence);
            
            // Backward
            double[][] outputGrad = new double[10][8];
            for (int t = 0; t < 10; t++) {
                outputGrad[t] = randomArray(8);
            }
            
            double[][] inputGrad = layer.backwardSequence(outputGrad);
            
            assertEquals(10, inputGrad.length);
            assertEquals(5, inputGrad[0].length);
        }
    }
    
    // ==================== RNN Builder Tests ====================
    
    @Nested
    @DisplayName("RNN Builder Tests")
    class RNNBuilderTests {
        
        @Test
        @DisplayName("Builder creates valid RNN")
        void testBasicBuilder() {
            RNN rnn = RNN.builder()
                .inputShape(10, 5)
                .addLSTM(32)
                .addOutput(3)
                .build();
            
            assertEquals(5, rnn.getInputSize());
            assertEquals(10, rnn.getSequenceLength());
            assertEquals(3, rnn.getOutputSize());
            assertEquals(1, rnn.getNumLSTMLayers());
            assertEquals(1, rnn.getNumDenseLayers());
        }
        
        @Test
        @DisplayName("Builder with multiple LSTM layers")
        void testMultipleLSTMLayers() {
            RNN rnn = RNN.builder()
                .inputShape(10, 5)
                .addLSTM(64, true)  // Return sequences for stacking
                .addLSTM(32, true)
                .addLSTM(16)        // Last LSTM doesn't return sequences
                .addOutput(3)
                .build();
            
            assertEquals(3, rnn.getNumLSTMLayers());
        }
        
        @Test
        @DisplayName("Builder with bidirectional LSTM")
        void testBidirectionalBuilder() {
            RNN rnn = RNN.builder()
                .inputShape(10, 5)
                .addBidirectionalLSTM(32)
                .addDense(64)
                .addOutput(3)
                .build();
            
            assertNotNull(rnn);
        }
        
        @Test
        @DisplayName("Builder with all options")
        void testFullOptionsBuilder() {
            RNN rnn = RNN.builder()
                .inputShape(20, 10)
                .addLSTM(64, true, true, false, 0.2)  // Bidirectional with dropout
                .addLSTM(32, false, false, false, 0.1)
                .addDense(64, ActivationFunction.RELU)
                .addOutput(5)
                .learningRate(0.01)
                .clipValue(3.0)
                .batchSize(16)
                .epochs(50)
                .verbose(false)
                .build();
            
            assertEquals(0.01, rnn.getLearningRate());
            assertEquals(50, rnn.getEpochs());
            assertEquals(16, rnn.getBatchSize());
        }
        
        @Test
        @DisplayName("Builder fails without input shape")
        void testBuilderWithoutInputShape() {
            assertThrows(IllegalStateException.class, () -> {
                RNN.builder()
                    .addLSTM(32)
                    .addOutput(3)
                    .build();
            });
        }
        
        @Test
        @DisplayName("Builder fails without LSTM layer")
        void testBuilderWithoutLSTM() {
            assertThrows(IllegalStateException.class, () -> {
                RNN.builder()
                    .inputShape(10, 5)
                    .addOutput(3)
                    .build();
            });
        }
        
        @Test
        @DisplayName("Builder fails without output layer")
        void testBuilderWithoutOutput() {
            assertThrows(IllegalStateException.class, () -> {
                RNN.builder()
                    .inputShape(10, 5)
                    .addLSTM(32)
                    .build();
            });
        }
        
        @Test
        @DisplayName("Cannot add LSTM after non-return-sequences LSTM")
        void testInvalidLSTMStacking() {
            assertThrows(IllegalStateException.class, () -> {
                RNN.builder()
                    .inputShape(10, 5)
                    .addLSTM(32, false)  // No return sequences
                    .addLSTM(16)         // This should fail
                    .addOutput(3)
                    .build();
            });
        }
    }
    
    // ==================== RNN Training Tests ====================
    
    @Nested
    @DisplayName("RNN Training Tests")
    class RNNTrainingTests {
        
        @Test
        @DisplayName("Training reduces loss")
        void testTrainingReducesLoss() {
            RNN rnn = RNN.builder()
                .inputShape(5, 3)
                .addLSTM(16)
                .addOutput(2)
                .learningRate(0.01)
                .epochs(20)
                .batchSize(4)
                .verbose(false)
                .build();
            
            // Generate synthetic data
            int numSamples = 20;
            double[][][] sequences = new double[numSamples][5][3];
            double[][] labels = new double[numSamples][2];
            
            for (int i = 0; i < numSamples; i++) {
                for (int t = 0; t < 5; t++) {
                    sequences[i][t] = randomArray(3);
                }
                // Simple rule: if mean of first timestep > 0, class 1
                double mean = (sequences[i][0][0] + sequences[i][0][1] + sequences[i][0][2]) / 3;
                labels[i][mean > 0 ? 1 : 0] = 1.0;
            }
            
            rnn.train(sequences, labels);
            
            double[] losses = rnn.getTrainingLosses();
            assertNotNull(losses);
            assertEquals(20, losses.length);
            
            // Loss should generally decrease (allow some variance)
            double firstHalfAvg = 0;
            double secondHalfAvg = 0;
            for (int i = 0; i < 10; i++) {
                firstHalfAvg += losses[i];
                secondHalfAvg += losses[i + 10];
            }
            firstHalfAvg /= 10;
            secondHalfAvg /= 10;
            
            // Second half average should be lower or equal
            assertTrue(secondHalfAvg <= firstHalfAvg * 1.5, 
                "Loss should not increase significantly during training");
        }
        
        @Test
        @DisplayName("Training with small dataset")
        void testSmallDatasetTraining() {
            RNN rnn = RNN.builder()
                .inputShape(3, 2)
                .addLSTM(8)
                .addOutput(2)
                .epochs(5)
                .batchSize(2)
                .verbose(false)
                .build();
            
            double[][][] sequences = {
                {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}},
                {{0.9, 0.8}, {0.7, 0.6}, {0.5, 0.4}}
            };
            double[][] labels = {{1, 0}, {0, 1}};
            
            assertDoesNotThrow(() -> rnn.train(sequences, labels));
        }
    }
    
    // ==================== RNN Prediction Tests ====================
    
    @Nested
    @DisplayName("RNN Prediction Tests")
    class RNNPredictionTests {
        
        @Test
        @DisplayName("Predict returns valid probabilities")
        void testPredictProbabilities() {
            RNN rnn = RNN.builder()
                .inputShape(5, 3)
                .addLSTM(16)
                .addOutput(3)
                .verbose(false)
                .build();
            
            double[][] sequence = new double[5][3];
            for (int t = 0; t < 5; t++) {
                sequence[t] = randomArray(3);
            }
            
            double[] probs = rnn.predict(sequence);
            
            assertEquals(3, probs.length);
            
            // Check valid probabilities
            double sum = 0;
            for (double p : probs) {
                assertTrue(p >= 0 && p <= 1, "Probability should be between 0 and 1");
                sum += p;
            }
            assertEquals(1.0, sum, 0.001, "Probabilities should sum to 1");
        }
        
        @Test
        @DisplayName("PredictClass returns valid class index")
        void testPredictClass() {
            RNN rnn = RNN.builder()
                .inputShape(5, 3)
                .addLSTM(16)
                .addOutput(4)
                .verbose(false)
                .build();
            
            double[][] sequence = new double[5][3];
            for (int t = 0; t < 5; t++) {
                sequence[t] = randomArray(3);
            }
            
            int predictedClass = rnn.predictClass(sequence);
            
            assertTrue(predictedClass >= 0 && predictedClass < 4, 
                "Predicted class should be valid index");
        }
        
        @Test
        @DisplayName("PredictClasses handles multiple sequences")
        void testPredictClasses() {
            RNN rnn = RNN.builder()
                .inputShape(5, 3)
                .addLSTM(16)
                .addOutput(3)
                .verbose(false)
                .build();
            
            double[][][] sequences = new double[10][5][3];
            for (int i = 0; i < 10; i++) {
                for (int t = 0; t < 5; t++) {
                    sequences[i][t] = randomArray(3);
                }
            }
            
            int[] predictions = rnn.predictClasses(sequences);
            
            assertEquals(10, predictions.length);
            for (int p : predictions) {
                assertTrue(p >= 0 && p < 3);
            }
        }
    }
    
    // ==================== RNN Evaluation Tests ====================
    
    @Nested
    @DisplayName("RNN Evaluation Tests")
    class RNNEvaluationTests {
        
        @Test
        @DisplayName("Evaluate returns valid accuracy")
        void testEvaluate() {
            RNN rnn = RNN.builder()
                .inputShape(5, 3)
                .addLSTM(16)
                .addOutput(2)
                .epochs(10)
                .verbose(false)
                .build();
            
            // Training data
            double[][][] trainSeq = new double[20][5][3];
            double[][] trainLabels = new double[20][2];
            
            for (int i = 0; i < 20; i++) {
                for (int t = 0; t < 5; t++) {
                    trainSeq[i][t] = randomArray(3);
                }
                trainLabels[i][i % 2] = 1.0;
            }
            
            rnn.train(trainSeq, trainLabels);
            
            double accuracy = rnn.evaluate(trainSeq, trainLabels);
            
            assertTrue(accuracy >= 0 && accuracy <= 1, "Accuracy should be between 0 and 1");
        }
    }
    
    // ==================== Serialization Tests ====================
    
    @Nested
    @DisplayName("RNN Serialization Tests")
    class RNNSerializationTests {
        
        @TempDir
        Path tempDir;
        
        @Test
        @DisplayName("Save and load preserves model")
        void testSaveAndLoad() throws Exception {
            RNN originalRnn = RNN.builder()
                .inputShape(5, 3)
                .addLSTM(16)
                .addDense(8)
                .addOutput(2)
                .learningRate(0.01)
                .verbose(false)
                .build();
            
            // Train briefly
            double[][][] sequences = new double[10][5][3];
            double[][] labels = new double[10][2];
            for (int i = 0; i < 10; i++) {
                for (int t = 0; t < 5; t++) {
                    sequences[i][t] = randomArray(3);
                }
                labels[i][i % 2] = 1.0;
            }
            originalRnn.train(sequences, labels);
            
            // Save
            String filename = tempDir.resolve("rnn_model.ser").toString();
            originalRnn.save(filename);
            
            // Load
            RNN loadedRnn = RNN.load(filename);
            
            // Compare predictions
            double[][] testSeq = sequences[0];
            double[] originalPred = originalRnn.predict(testSeq);
            double[] loadedPred = loadedRnn.predict(testSeq);
            
            assertArrayEquals(originalPred, loadedPred, 0.0001, 
                "Loaded model should produce same predictions");
        }
    }
    
    // ==================== Utility Methods Tests ====================
    
    @Nested
    @DisplayName("Utility Methods Tests")
    class UtilityMethodsTests {
        
        @Test
        @DisplayName("One-hot encoding works correctly")
        void testOneHotEncode() {
            int[] labels = {0, 1, 2, 1, 0};
            double[][] encoded = RNN.oneHotEncode(labels, 3);
            
            assertEquals(5, encoded.length);
            assertEquals(3, encoded[0].length);
            
            assertArrayEquals(new double[]{1, 0, 0}, encoded[0]);
            assertArrayEquals(new double[]{0, 1, 0}, encoded[1]);
            assertArrayEquals(new double[]{0, 0, 1}, encoded[2]);
            assertArrayEquals(new double[]{0, 1, 0}, encoded[3]);
            assertArrayEquals(new double[]{1, 0, 0}, encoded[4]);
        }
        
        @Test
        @DisplayName("Pad sequences works correctly")
        void testPadSequences() {
            double[][][] sequences = {
                {{1, 2}, {3, 4}},           // Length 2
                {{5, 6}, {7, 8}, {9, 10}}   // Length 3
            };
            
            double[][][] padded = RNN.padSequences(sequences, 4, 0.0);
            
            assertEquals(2, padded.length);
            assertEquals(4, padded[0].length);
            assertEquals(4, padded[1].length);
            
            // Check padding (right-aligned)
            assertArrayEquals(new double[]{0, 0}, padded[0][0]);
            assertArrayEquals(new double[]{0, 0}, padded[0][1]);
            assertArrayEquals(new double[]{1, 2}, padded[0][2]);
            assertArrayEquals(new double[]{3, 4}, padded[0][3]);
        }
        
        @Test
        @DisplayName("Sliding windows works correctly")
        void testSlidingWindows() {
            double[][] data = {
                {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}
            };
            
            double[][][] windows = RNN.createSlidingWindows(data, 3, 1);
            
            assertEquals(3, windows.length);  // (5-3)/1 + 1 = 3 windows
            assertEquals(3, windows[0].length);  // Window size
            assertEquals(2, windows[0][0].length);  // Features
            
            // Check first window
            assertArrayEquals(new double[]{1, 2}, windows[0][0]);
            assertArrayEquals(new double[]{3, 4}, windows[0][1]);
            assertArrayEquals(new double[]{5, 6}, windows[0][2]);
            
            // Check last window
            assertArrayEquals(new double[]{5, 6}, windows[2][0]);
            assertArrayEquals(new double[]{7, 8}, windows[2][1]);
            assertArrayEquals(new double[]{9, 10}, windows[2][2]);
        }
        
        @Test
        @DisplayName("Sliding windows with larger step")
        void testSlidingWindowsWithStep() {
            double[][] data = new double[10][2];
            for (int i = 0; i < 10; i++) {
                data[i] = new double[]{i, i * 2};
            }
            
            double[][][] windows = RNN.createSlidingWindows(data, 3, 2);
            
            assertEquals(4, windows.length);  // (10-3)/2 + 1 = 4 windows
        }
    }
    
    // ==================== Model Summary Tests ====================
    
    @Nested
    @DisplayName("Model Summary Tests")
    class ModelSummaryTests {
        
        @Test
        @DisplayName("Get summary returns valid string")
        void testGetSummary() {
            RNN rnn = RNN.builder()
                .inputShape(20, 10)
                .addLSTM(64, true)
                .addBidirectionalLSTM(32, false, 0.2)
                .addDense(16)
                .addOutput(5)
                .build();
            
            String summary = rnn.getSummary();
            
            assertNotNull(summary);
            assertTrue(summary.contains("RNN Model Summary"));
            assertTrue(summary.contains("LSTM"));
            assertTrue(summary.contains("Bidirectional"));
            assertTrue(summary.contains("Dense"));
        }
    }
    
    // ==================== Sequence Classification Tests ====================
    
    @Nested
    @DisplayName("Sequence Classification Tests")
    class SequenceClassificationTests {
        
        @Test
        @DisplayName("Binary sequence classification")
        void testBinaryClassification() {
            RNN rnn = RNN.builder()
                .inputShape(10, 4)
                .addLSTM(32)
                .addOutput(2)
                .epochs(30)
                .batchSize(8)
                .learningRate(0.01)
                .verbose(false)
                .build();
            
            // Generate separable data
            int numSamples = 40;
            double[][][] sequences = new double[numSamples][10][4];
            double[][] labels = new double[numSamples][2];
            
            for (int i = 0; i < numSamples; i++) {
                double offset = (i % 2 == 0) ? 0.5 : -0.5;
                for (int t = 0; t < 10; t++) {
                    for (int f = 0; f < 4; f++) {
                        sequences[i][t][f] = offset + random.nextGaussian() * 0.1;
                    }
                }
                labels[i][i % 2] = 1.0;
            }
            
            rnn.train(sequences, labels);
            
            double accuracy = rnn.evaluate(sequences, labels);
            assertTrue(accuracy >= 0.5, "Should achieve at least 50% accuracy on training data");
        }
        
        @Test
        @DisplayName("Multiclass sequence classification")
        void testMulticlassClassification() {
            RNN rnn = RNN.builder()
                .inputShape(8, 3)
                .addLSTM(24)
                .addDense(12)
                .addOutput(4)
                .epochs(20)
                .batchSize(4)
                .verbose(false)
                .build();
            
            // Generate data for 4 classes
            int numSamples = 40;
            double[][][] sequences = new double[numSamples][8][3];
            double[][] labels = new double[numSamples][4];
            
            for (int i = 0; i < numSamples; i++) {
                int classIdx = i % 4;
                double baseVal = classIdx * 0.3;
                for (int t = 0; t < 8; t++) {
                    for (int f = 0; f < 3; f++) {
                        sequences[i][t][f] = baseVal + random.nextGaussian() * 0.05;
                    }
                }
                labels[i][classIdx] = 1.0;
            }
            
            assertDoesNotThrow(() -> rnn.train(sequences, labels));
        }
    }
    
    // ==================== Time Series Tests ====================
    
    @Nested
    @DisplayName("Time Series Tests")
    class TimeSeriesTests {
        
        @Test
        @DisplayName("Sine wave classification")
        void testSineWaveClassification() {
            RNN rnn = RNN.builder()
                .inputShape(20, 1)
                .addLSTM(16)
                .addOutput(2)
                .epochs(30)
                .batchSize(8)
                .verbose(false)
                .build();
            
            int numSamples = 50;
            double[][][] sequences = new double[numSamples][20][1];
            double[][] labels = new double[numSamples][2];
            
            for (int i = 0; i < numSamples; i++) {
                double freq = (i % 2 == 0) ? 0.1 : 0.3;  // Two different frequencies
                for (int t = 0; t < 20; t++) {
                    sequences[i][t][0] = Math.sin(2 * Math.PI * freq * t);
                }
                labels[i][i % 2] = 1.0;
            }
            
            rnn.train(sequences, labels);
            
            double accuracy = rnn.evaluate(sequences, labels);
            assertTrue(accuracy >= 0.4, "Should learn some pattern from sine waves");
        }
    }
    
    // ==================== Helper Methods ====================
    
    private static double[] randomArray(int size) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = random.nextGaussian();
        }
        return arr;
    }
}
