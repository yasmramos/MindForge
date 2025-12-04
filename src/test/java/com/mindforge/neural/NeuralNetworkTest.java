package com.mindforge.neural;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Neural Network Tests")
class NeuralNetworkTest {
    
    @Nested
    @DisplayName("Activation Function Tests")
    class ActivationFunctionTests {
        
        @Test
        @DisplayName("Sigmoid should return values between 0 and 1")
        void testSigmoid() {
            double[] inputs = {-10.0, -1.0, 0.0, 1.0, 10.0};
            
            for (double x : inputs) {
                double output = ActivationFunction.SIGMOID.apply(x);
                assertTrue(output >= 0 && output <= 1, "Sigmoid output should be between 0 and 1");
            }
            assertEquals(0.5, ActivationFunction.SIGMOID.apply(0.0), 0.001, "Sigmoid(0) should be 0.5");
        }
        
        @Test
        @DisplayName("ReLU should return max(0, x)")
        void testReLU() {
            assertEquals(0.0, ActivationFunction.RELU.apply(-2.0), 0.001);
            assertEquals(0.0, ActivationFunction.RELU.apply(-1.0), 0.001);
            assertEquals(0.0, ActivationFunction.RELU.apply(0.0), 0.001);
            assertEquals(1.0, ActivationFunction.RELU.apply(1.0), 0.001);
            assertEquals(2.0, ActivationFunction.RELU.apply(2.0), 0.001);
        }
        
        @Test
        @DisplayName("Tanh should return values between -1 and 1")
        void testTanh() {
            double[] inputs = {-10.0, 0.0, 10.0};
            
            for (double x : inputs) {
                double output = ActivationFunction.TANH.apply(x);
                assertTrue(output >= -1 && output <= 1, "Tanh output should be between -1 and 1");
            }
            assertEquals(0.0, ActivationFunction.TANH.apply(0.0), 0.001, "Tanh(0) should be 0");
        }
        
        @Test
        @DisplayName("Leaky ReLU should allow small negative values")
        void testLeakyReLU() {
            double negOutput = ActivationFunction.LEAKY_RELU.apply(-2.0);
            assertTrue(negOutput < 0, "Leaky ReLU should be negative for negative input");
            assertEquals(0.0, ActivationFunction.LEAKY_RELU.apply(0.0), 0.001);
            assertEquals(2.0, ActivationFunction.LEAKY_RELU.apply(2.0), 0.001);
        }
        
        @Test
        @DisplayName("ELU should handle negative values smoothly")
        void testELU() {
            double negOutput = ActivationFunction.ELU.apply(-2.0);
            assertTrue(negOutput < 0, "ELU should be negative for negative inputs");
            assertEquals(0.0, ActivationFunction.ELU.apply(0.0), 0.001);
            assertEquals(2.0, ActivationFunction.ELU.apply(2.0), 0.001);
        }
        
        @Test
        @DisplayName("Sigmoid derivative should be correct")
        void testSigmoidDerivative() {
            // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
            assertEquals(0.25, ActivationFunction.SIGMOID.derivative(0.0), 0.001);
        }
        
        @Test
        @DisplayName("ReLU derivative should be correct")
        void testReLUDerivative() {
            assertEquals(0.0, ActivationFunction.RELU.derivative(-1.0), 0.001);
            assertEquals(1.0, ActivationFunction.RELU.derivative(1.0), 0.001);
        }
        
        @Test
        @DisplayName("All activation functions should have derivatives")
        void testAllDerivatives() {
            for (ActivationFunction af : ActivationFunction.values()) {
                assertDoesNotThrow(() -> af.derivative(0.5), 
                    "Derivative should be defined for " + af.name());
            }
        }
    }
    
    @Nested
    @DisplayName("Dense Layer Tests")
    class DenseLayerTests {
        
        @Test
        @DisplayName("Dense layer should produce correct output size")
        void testDenseLayerOutputSize() {
            DenseLayer layer = new DenseLayer(4, 3, ActivationFunction.RELU);
            double[] input = {1.0, 2.0, 3.0, 4.0};
            double[] output = layer.forward(input);
            
            assertEquals(3, output.length, "Output size should match layer output size");
        }
        
        @Test
        @DisplayName("Dense layer should apply activation function")
        void testDenseLayerActivation() {
            DenseLayer layer = new DenseLayer(2, 2, ActivationFunction.RELU);
            double[] input = {1.0, 1.0};
            double[] output = layer.forward(input);
            
            for (double val : output) {
                assertTrue(val >= 0, "ReLU output should be non-negative");
            }
        }
        
        @Test
        @DisplayName("Dense layer backward should return gradient")
        void testDenseLayerBackward() {
            DenseLayer layer = new DenseLayer(3, 2, ActivationFunction.SIGMOID);
            double[] input = {1.0, 2.0, 3.0};
            layer.forward(input);
            
            double[] outputGradient = {0.1, 0.2};
            double[] inputGradient = layer.backward(outputGradient, 0.01);
            
            assertEquals(3, inputGradient.length, "Input gradient should match input size");
        }
        
        @Test
        @DisplayName("Dense layer should be reproducible with seed")
        void testDenseLayerSeed() {
            DenseLayer layer1 = new DenseLayer(2, 3, ActivationFunction.RELU, 42);
            DenseLayer layer2 = new DenseLayer(2, 3, ActivationFunction.RELU, 42);
            
            double[] input = {1.0, 2.0};
            double[] output1 = layer1.forward(input);
            double[] output2 = layer2.forward(input);
            
            assertArrayEquals(output1, output2, 0.001, "Same seed should produce same output");
        }
    }
    
    @Nested
    @DisplayName("Neural Network Tests")
    class NeuralNetworkMainTests {
        
        private NeuralNetwork network;
        
        @BeforeEach
        void setUp() {
            network = new NeuralNetwork(0.1, 50, 4);
            network.addLayer(new DenseLayer(2, 4, ActivationFunction.RELU));
            network.addLayer(new DenseLayer(4, 2, ActivationFunction.SOFTMAX));
        }
        
        @Test
        @DisplayName("Network should predict after training")
        void testNetworkPredict() {
            double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            int[] y = {0, 1, 1, 0};
            
            network.setVerbose(false);
            network.fit(X, y);
            
            int[] predictions = network.predict(X);
            assertNotNull(predictions, "Predictions should not be null");
            assertEquals(4, predictions.length, "Should have prediction for each sample");
        }
        
        @Test
        @DisplayName("Network should add layers with addDenseLayer")
        void testAddDenseLayer() {
            NeuralNetwork net = new NeuralNetwork();
            net.addDenseLayer(2, 3, ActivationFunction.RELU);
            net.addDenseLayer(3, 2, ActivationFunction.SOFTMAX);
            
            double[] input = {1.0, 2.0};
            double[] output = net.forward(input);
            
            assertEquals(2, output.length);
        }
        
        @Test
        @DisplayName("Network should return probabilities with predictProba")
        void testPredictProba() {
            double[][] X = {{0, 0}, {0, 1}};
            int[] y = {0, 1};
            
            network.setVerbose(false);
            network.fit(X, y);
            
            double[][] proba = network.predictProba(X);
            
            assertEquals(2, proba.length);
            assertEquals(2, proba[0].length);
            
            // Probabilities should sum to approximately 1
            for (double[] p : proba) {
                double sum = 0;
                for (double val : p) {
                    sum += val;
                    assertTrue(val >= 0 && val <= 1, "Probability should be between 0 and 1");
                }
                assertEquals(1.0, sum, 0.01, "Probabilities should sum to 1");
            }
        }
        
        @Test
        @DisplayName("Network should calculate score")
        void testScore() {
            double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            int[] y = {0, 1, 1, 0};
            
            network.setVerbose(false);
            network.fit(X, y);
            
            double score = network.score(X, y);
            
            assertTrue(score >= 0 && score <= 1, "Score should be between 0 and 1");
        }
        
        @Test
        @DisplayName("Network should get training loss history")
        void testTrainingLoss() {
            double[][] X = {{0, 0}, {0, 1}};
            int[] y = {0, 1};
            
            network.setVerbose(false);
            network.fit(X, y);
            
            java.util.List<Double> losses = network.getTrainingLoss();
            
            assertNotNull(losses);
            assertFalse(losses.isEmpty(), "Should have recorded losses");
        }
    }
    
    @Nested
    @DisplayName("Dropout Layer Tests")
    class DropoutLayerTests {
        
        @Test
        @DisplayName("Dropout should not change size")
        void testDropoutSize() {
            DropoutLayer dropout = new DropoutLayer(4, 0.5);
            double[] input = {1.0, 2.0, 3.0, 4.0};
            double[] output = dropout.forward(input);
            
            assertEquals(input.length, output.length, "Dropout should preserve input size");
        }
        
        @Test
        @DisplayName("Dropout should pass through during inference")
        void testDropoutInference() {
            DropoutLayer dropout = new DropoutLayer(3, 0.5);
            dropout.setTraining(false);
            double[] input = {1.0, 2.0, 3.0};
            double[] output = dropout.forward(input);
            
            assertArrayEquals(input, output, 0.001, "Inference should pass through unchanged");
        }
        
        @Test
        @DisplayName("Dropout should return correct dropout rate")
        void testDropoutRate() {
            DropoutLayer dropout = new DropoutLayer(4, 0.3);
            assertEquals(0.3, dropout.getDropoutRate(), 0.001);
        }
        
        @Test
        @DisplayName("Dropout backward should return gradient")
        void testDropoutBackward() {
            DropoutLayer dropout = new DropoutLayer(3, 0.5, 42);
            double[] input = {1.0, 2.0, 3.0};
            dropout.forward(input);
            
            double[] gradOutput = {0.1, 0.2, 0.3};
            double[] gradInput = dropout.backward(gradOutput, 0.01);
            
            assertEquals(3, gradInput.length);
        }
    }
    
    @Nested
    @DisplayName("Batch Normalization Tests")
    class BatchNormTests {
        
        @Test
        @DisplayName("BatchNorm should preserve size")
        void testBatchNormForward() {
            BatchNormLayer bn = new BatchNormLayer(3);
            double[] input = {10.0, 20.0, 30.0};
            double[] output = bn.forward(input);
            
            assertEquals(3, output.length, "Output size should match input size");
        }
        
        @Test
        @DisplayName("BatchNorm backward should return gradient")
        void testBatchNormBackward() {
            BatchNormLayer bn = new BatchNormLayer(3);
            double[] input = {1.0, 2.0, 3.0};
            bn.forward(input);
            
            double[] outputGradient = {0.1, 0.2, 0.3};
            double[] inputGradient = bn.backward(outputGradient, 0.01);
            
            assertEquals(3, inputGradient.length);
        }
    }
    
    @Nested
    @DisplayName("Regression Tests")
    class RegressionTests {
        
        @Test
        @DisplayName("Network should train for regression")
        void testRegressionTraining() {
            NeuralNetwork net = new NeuralNetwork(0.01, 10, 2);
            net.addLayer(new DenseLayer(2, 4, ActivationFunction.RELU));
            net.addLayer(new DenseLayer(4, 1, ActivationFunction.LINEAR));
            
            double[][] X = {{1, 2}, {3, 4}, {5, 6}};
            double[] y = {3, 7, 11};
            
            net.setVerbose(false);
            assertDoesNotThrow(() -> net.fitRegression(X, y));
        }
        
        @Test
        @DisplayName("Network should predict regression values")
        void testRegressionPredict() {
            NeuralNetwork net = new NeuralNetwork(0.01, 10, 2);
            net.addLayer(new DenseLayer(2, 4, ActivationFunction.RELU));
            net.addLayer(new DenseLayer(4, 1, ActivationFunction.LINEAR));
            
            double[][] X = {{1, 2}, {3, 4}};
            double[] y = {3, 7};
            
            net.setVerbose(false);
            net.fitRegression(X, y);
            
            double[] predictions = net.predictRegression(X);
            assertEquals(2, predictions.length);
        }
    }
    
    @Nested
    @DisplayName("Edge Cases")
    class EdgeCases {
        
        @Test
        @DisplayName("Should handle single neuron layers")
        void testSingleNeuronLayer() {
            DenseLayer layer = new DenseLayer(1, 1, ActivationFunction.SIGMOID);
            double[] input = {0.5};
            double[] output = layer.forward(input);
            
            assertEquals(1, output.length);
            assertTrue(output[0] >= 0 && output[0] <= 1);
        }
        
        @Test
        @DisplayName("Should handle large networks")
        void testLargeNetwork() {
            NeuralNetwork net = new NeuralNetwork();
            net.addLayer(new DenseLayer(10, 64, ActivationFunction.RELU));
            net.addLayer(new DenseLayer(64, 32, ActivationFunction.RELU));
            net.addLayer(new DenseLayer(32, 16, ActivationFunction.RELU));
            net.addLayer(new DenseLayer(16, 2, ActivationFunction.SOFTMAX));
            
            double[] input = new double[10];
            for (int i = 0; i < 10; i++) input[i] = i * 0.1;
            
            double[] output = net.forward(input);
            assertEquals(2, output.length);
        }
        
        @Test
        @DisplayName("Should set hyperparameters")
        void testSetHyperparameters() {
            NeuralNetwork net = new NeuralNetwork();
            net.setLearningRate(0.001);
            net.setEpochs(200);
            net.setBatchSize(16);
            
            // Should not throw
            assertNotNull(net);
        }
        
        @Test
        @DisplayName("Should get layers")
        void testGetLayers() {
            NeuralNetwork net = new NeuralNetwork();
            net.addLayer(new DenseLayer(2, 3, ActivationFunction.RELU));
            net.addLayer(new DenseLayer(3, 1, ActivationFunction.SIGMOID));
            
            java.util.List<Layer> layers = net.getLayers();
            
            assertEquals(2, layers.size());
        }
    }
}
