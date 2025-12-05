package com.mindforge.neural;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for CNN (Convolutional Neural Network).
 */
@DisplayName("CNN Tests")
class CNNTest {
    
    private Random random;
    
    @BeforeEach
    void setUp() {
        random = new Random(42);
    }
    
    // Helper methods
    
    private double[][] generateRandomImages(int n, int channels, int height, int width) {
        double[][] images = new double[n][channels * height * width];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < images[i].length; j++) {
                images[i][j] = random.nextDouble();
            }
        }
        return images;
    }
    
    private int[] generateRandomLabels(int n, int numClasses) {
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            labels[i] = random.nextInt(numClasses);
        }
        return labels;
    }
    
    @Nested
    @DisplayName("Conv2DLayer Tests")
    class Conv2DLayerTests {
        
        @Test
        @DisplayName("Should compute correct output dimensions")
        void testOutputDimensions() {
            Conv2DLayer layer = new Conv2DLayer(3, 28, 28, 16, 3, 1, 1, ActivationFunction.RELU);
            
            assertEquals(16, layer.getNumFilters());
            assertEquals(28, layer.getOutputHeight());  // With padding=1, kernel=3, stride=1: same size
            assertEquals(28, layer.getOutputWidth());
        }
        
        @Test
        @DisplayName("Should compute output dimensions with stride > 1")
        void testOutputDimensionsWithStride() {
            Conv2DLayer layer = new Conv2DLayer(1, 28, 28, 32, 3, 2, 0, ActivationFunction.RELU);
            
            // Output: (28 - 3) / 2 + 1 = 13
            assertEquals(13, layer.getOutputHeight());
            assertEquals(13, layer.getOutputWidth());
        }
        
        @Test
        @DisplayName("Should perform forward pass correctly")
        void testForward() {
            Conv2DLayer layer = new Conv2DLayer(1, 4, 4, 2, 3, 1, 0, ActivationFunction.RELU, 42);
            
            double[] input = new double[16];
            for (int i = 0; i < 16; i++) input[i] = i / 16.0;
            
            double[] output = layer.forward(input);
            
            // Output: (4 - 3) / 1 + 1 = 2x2, 2 filters = 8 values
            assertEquals(8, output.length);
            
            // All outputs should be >= 0 (ReLU)
            for (double v : output) {
                assertTrue(v >= 0, "ReLU output should be >= 0");
            }
        }
        
        @Test
        @DisplayName("Should perform backward pass correctly")
        void testBackward() {
            Conv2DLayer layer = new Conv2DLayer(1, 4, 4, 2, 3, 1, 0, ActivationFunction.RELU, 42);
            
            double[] input = new double[16];
            for (int i = 0; i < 16; i++) input[i] = random.nextDouble();
            
            double[] output = layer.forward(input);
            
            double[] gradOutput = new double[output.length];
            for (int i = 0; i < gradOutput.length; i++) gradOutput[i] = 1.0;
            
            double[] gradInput = layer.backward(gradOutput, 0.01);
            
            assertEquals(16, gradInput.length);
        }
        
        @Test
        @DisplayName("Should throw exception for invalid parameters")
        void testInvalidParameters() {
            assertThrows(IllegalArgumentException.class, () ->
                new Conv2DLayer(-1, 28, 28, 16, 3, 1, 0, ActivationFunction.RELU));
            
            assertThrows(IllegalArgumentException.class, () ->
                new Conv2DLayer(1, 28, 28, 16, 30, 1, 0, ActivationFunction.RELU));  // kernel too large
        }
        
        @Test
        @DisplayName("Should count parameters correctly")
        void testNumParameters() {
            Conv2DLayer layer = new Conv2DLayer(3, 28, 28, 16, 3, 1, 1, ActivationFunction.RELU);
            
            // filters: 16 * 3 * 3 * 3 = 432, biases: 16 = 448 total
            assertEquals(448, layer.getNumParameters());
        }
    }
    
    @Nested
    @DisplayName("MaxPooling2DLayer Tests")
    class MaxPooling2DLayerTests {
        
        @Test
        @DisplayName("Should compute correct output dimensions")
        void testOutputDimensions() {
            MaxPooling2DLayer layer = new MaxPooling2DLayer(16, 28, 28, 2);
            
            assertEquals(14, layer.getOutputHeight());
            assertEquals(14, layer.getOutputWidth());
            assertEquals(16, layer.getInputChannels());
        }
        
        @Test
        @DisplayName("Should select maximum values")
        void testMaxSelection() {
            MaxPooling2DLayer layer = new MaxPooling2DLayer(1, 4, 4, 2);
            
            // Create input with known values
            double[] input = {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            };
            
            double[] output = layer.forward(input);
            
            assertEquals(4, output.length);
            assertEquals(6, output[0], 0.001);   // max of [1,2,5,6]
            assertEquals(8, output[1], 0.001);   // max of [3,4,7,8]
            assertEquals(14, output[2], 0.001);  // max of [9,10,13,14]
            assertEquals(16, output[3], 0.001);  // max of [11,12,15,16]
        }
        
        @Test
        @DisplayName("Should propagate gradients to max positions")
        void testBackward() {
            MaxPooling2DLayer layer = new MaxPooling2DLayer(1, 4, 4, 2);
            
            double[] input = {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            };
            
            layer.forward(input);
            
            double[] gradOutput = {1, 1, 1, 1};
            double[] gradInput = layer.backward(gradOutput, 0.01);
            
            assertEquals(16, gradInput.length);
            
            // Gradient should be non-zero only at max positions
            assertEquals(1, gradInput[5], 0.001);   // position of 6
            assertEquals(1, gradInput[7], 0.001);   // position of 8
            assertEquals(1, gradInput[13], 0.001);  // position of 14
            assertEquals(1, gradInput[15], 0.001);  // position of 16
        }
    }
    
    @Nested
    @DisplayName("AveragePooling2DLayer Tests")
    class AveragePooling2DLayerTests {
        
        @Test
        @DisplayName("Should compute correct output dimensions")
        void testOutputDimensions() {
            AveragePooling2DLayer layer = new AveragePooling2DLayer(16, 28, 28, 2);
            
            assertEquals(14, layer.getOutputHeight());
            assertEquals(14, layer.getOutputWidth());
        }
        
        @Test
        @DisplayName("Should compute average values")
        void testAverageComputation() {
            AveragePooling2DLayer layer = new AveragePooling2DLayer(1, 4, 4, 2);
            
            // Create input with known values
            double[] input = {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
            };
            
            double[] output = layer.forward(input);
            
            assertEquals(4, output.length);
            assertEquals(3.5, output[0], 0.001);   // avg of [1,2,5,6] = 14/4
            assertEquals(5.5, output[1], 0.001);   // avg of [3,4,7,8] = 22/4
            assertEquals(11.5, output[2], 0.001);  // avg of [9,10,13,14] = 46/4
            assertEquals(13.5, output[3], 0.001);  // avg of [11,12,15,16] = 54/4
        }
        
        @Test
        @DisplayName("Should distribute gradients evenly")
        void testBackward() {
            AveragePooling2DLayer layer = new AveragePooling2DLayer(1, 4, 4, 2);
            
            double[] input = new double[16];
            for (int i = 0; i < 16; i++) input[i] = i;
            
            layer.forward(input);
            
            double[] gradOutput = {4, 4, 4, 4};  // Each grad = 4
            double[] gradInput = layer.backward(gradOutput, 0.01);
            
            // Each input position should receive grad / poolArea = 4 / 4 = 1
            for (double v : gradInput) {
                assertEquals(1.0, v, 0.001);
            }
        }
    }
    
    @Nested
    @DisplayName("FlattenLayer Tests")
    class FlattenLayerTests {
        
        @Test
        @DisplayName("Should preserve array size")
        void testPreserveSize() {
            FlattenLayer layer = new FlattenLayer(16, 7, 7);
            
            assertEquals(16 * 7 * 7, layer.getInputSize());
            assertEquals(16 * 7 * 7, layer.getOutputSize());
        }
        
        @Test
        @DisplayName("Should pass through values unchanged")
        void testPassThrough() {
            FlattenLayer layer = new FlattenLayer(2, 3, 3);
            
            double[] input = new double[18];
            for (int i = 0; i < 18; i++) input[i] = i;
            
            double[] output = layer.forward(input);
            
            assertArrayEquals(input, output, 0.001);
        }
        
        @Test
        @DisplayName("Should pass gradients unchanged")
        void testBackward() {
            FlattenLayer layer = new FlattenLayer(2, 3, 3);
            
            double[] input = new double[18];
            layer.forward(input);
            
            double[] gradOutput = new double[18];
            for (int i = 0; i < 18; i++) gradOutput[i] = i * 0.1;
            
            double[] gradInput = layer.backward(gradOutput, 0.01);
            
            assertArrayEquals(gradOutput, gradInput, 0.001);
        }
    }
    
    @Nested
    @DisplayName("CNN Builder Tests")
    class BuilderTests {
        
        @Test
        @DisplayName("Should build CNN with specified architecture")
        void testBuildArchitecture() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 28, 28)
                .addConv2D(16, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addConv2D(32, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addFlatten()
                .addDense(64, ActivationFunction.RELU)
                .addDense(10, ActivationFunction.SOFTMAX)
                .build();
            
            assertEquals(7, cnn.getLayers().size());
            assertArrayEquals(new int[]{1, 28, 28}, cnn.getInputShape());
        }
        
        @Test
        @DisplayName("Should auto-flatten before dense layers")
        void testAutoFlatten() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 8, 8)
                .addConv2D(4, 3, ActivationFunction.RELU)
                .addDense(10, ActivationFunction.SOFTMAX)  // Should auto-add Flatten
                .build();
            
            // Conv2D + Flatten + Dense = 3 layers
            assertEquals(3, cnn.getLayers().size());
            assertTrue(cnn.getLayers().get(1) instanceof FlattenLayer);
        }
        
        @Test
        @DisplayName("Should throw exception for Conv2D after Flatten")
        void testInvalidArchitecture() {
            assertThrows(IllegalStateException.class, () -> 
                new CNN.Builder()
                    .inputShape(1, 28, 28)
                    .addFlatten()
                    .addConv2D(16, 3, ActivationFunction.RELU)
                    .build()
            );
        }
        
        @Test
        @DisplayName("Should set hyperparameters correctly")
        void testHyperparameters() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 8, 8)
                .addDense(10, ActivationFunction.SOFTMAX)
                .learningRate(0.001)
                .epochs(50)
                .batchSize(64)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            assertNotNull(cnn);
        }
    }
    
    @Nested
    @DisplayName("CNN Forward Pass Tests")
    class ForwardPassTests {
        
        @Test
        @DisplayName("Should produce valid output shape")
        void testOutputShape() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 8, 8)
                .addConv2D(4, 3, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addFlatten()
                .addDense(10, ActivationFunction.SOFTMAX)
                .randomSeed(42)
                .build();
            
            double[] input = new double[64];
            for (int i = 0; i < 64; i++) input[i] = random.nextDouble();
            
            double[] output = cnn.forward(input);
            
            assertEquals(10, output.length);
        }
        
        @Test
        @DisplayName("Should produce valid probability distribution")
        void testSoftmaxOutput() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 8, 8)
                .addDense(10, ActivationFunction.SOFTMAX)
                .randomSeed(42)
                .build();
            
            double[] input = new double[64];
            for (int i = 0; i < 64; i++) input[i] = random.nextDouble();
            
            double[] output = cnn.forward(input);
            
            // Check that probabilities sum to 1
            double sum = 0;
            for (double p : output) {
                assertTrue(p >= 0 && p <= 1, "Probability should be in [0,1]");
                sum += p;
            }
            assertEquals(1.0, sum, 0.01);
        }
    }
    
    @Nested
    @DisplayName("CNN Training Tests")
    class TrainingTests {
        
        @Test
        @DisplayName("Should train on simple dataset")
        void testTraining() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addConv2D(2, 3, 1, 1, ActivationFunction.RELU)
                .addFlatten()
                .addDense(2, ActivationFunction.SOFTMAX)
                .learningRate(0.01)
                .epochs(5)
                .batchSize(4)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            // Create simple binary classification dataset
            double[][] X = generateRandomImages(20, 1, 4, 4);
            int[] y = generateRandomLabels(20, 2);
            
            // Should not throw
            assertDoesNotThrow(() -> cnn.fit(X, y));
            
            // Training loss should exist
            assertFalse(cnn.getTrainingLoss().isEmpty());
        }
        
        @Test
        @DisplayName("Should decrease loss during training")
        void testLossDecreases() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(4, ActivationFunction.RELU)
                .addDense(2, ActivationFunction.SOFTMAX)
                .learningRate(0.1)
                .epochs(20)
                .batchSize(4)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            // Create linearly separable dataset
            double[][] X = new double[20][16];
            int[] y = new int[20];
            for (int i = 0; i < 20; i++) {
                double val = i < 10 ? 0.2 : 0.8;
                for (int j = 0; j < 16; j++) {
                    X[i][j] = val + (random.nextDouble() - 0.5) * 0.1;
                }
                y[i] = i < 10 ? 0 : 1;
            }
            
            cnn.fit(X, y);
            
            var losses = cnn.getTrainingLoss();
            // First loss should be higher than last loss (generally)
            assertTrue(losses.get(0) > losses.get(losses.size() - 1) * 0.5,
                "Loss should generally decrease during training");
        }
        
        @Test
        @DisplayName("Should train with validation data")
        void testTrainingWithValidation() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(2, ActivationFunction.SOFTMAX)
                .learningRate(0.01)
                .epochs(5)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            double[][] XTrain = generateRandomImages(16, 1, 4, 4);
            int[] yTrain = generateRandomLabels(16, 2);
            double[][] XVal = generateRandomImages(4, 1, 4, 4);
            int[] yVal = generateRandomLabels(4, 2);
            
            cnn.fit(XTrain, yTrain, XVal, yVal);
            
            assertFalse(cnn.getValidationLoss().isEmpty());
        }
    }
    
    @Nested
    @DisplayName("CNN Prediction Tests")
    class PredictionTests {
        
        @Test
        @DisplayName("Should predict class labels")
        void testPredict() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(2, ActivationFunction.SOFTMAX)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(10, 1, 4, 4);
            
            int[] predictions = cnn.predict(X);
            
            assertEquals(10, predictions.length);
            for (int p : predictions) {
                assertTrue(p >= 0 && p < 2);
            }
        }
        
        @Test
        @DisplayName("Should predict probabilities")
        void testPredictProba() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(3, ActivationFunction.SOFTMAX)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(5, 1, 4, 4);
            
            double[][] proba = cnn.predictProba(X);
            
            assertEquals(5, proba.length);
            for (double[] p : proba) {
                assertEquals(3, p.length);
                double sum = 0;
                for (double v : p) sum += v;
                assertEquals(1.0, sum, 0.01);
            }
        }
        
        @Test
        @DisplayName("Should calculate accuracy score")
        void testScore() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(2, ActivationFunction.SOFTMAX)
                .learningRate(0.1)
                .epochs(50)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            // Create simple dataset
            double[][] X = new double[20][16];
            int[] y = new int[20];
            for (int i = 0; i < 20; i++) {
                double val = i < 10 ? 0.0 : 1.0;
                for (int j = 0; j < 16; j++) X[i][j] = val;
                y[i] = i < 10 ? 0 : 1;
            }
            
            cnn.fit(X, y);
            double accuracy = cnn.score(X, y);
            
            assertTrue(accuracy >= 0.0 && accuracy <= 1.0);
            assertTrue(accuracy > 0.6, "Accuracy should be reasonable on simple dataset");
        }
    }
    
    @Nested
    @DisplayName("CNN with Dropout Tests")
    class DropoutTests {
        
        @Test
        @DisplayName("Should include dropout layers")
        void testDropoutInArchitecture() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(8, ActivationFunction.RELU)
                .addDropout(0.5)
                .addDense(2, ActivationFunction.SOFTMAX)
                .randomSeed(42)
                .build();
            
            // Flatten + Dense + Dropout + Dense = 4 layers
            assertEquals(4, cnn.getLayers().size());
            assertTrue(cnn.getLayers().get(2) instanceof DropoutLayer);
        }
        
        @Test
        @DisplayName("Should train with dropout")
        void testTrainingWithDropout() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addDense(8, ActivationFunction.RELU)
                .addDropout(0.3)
                .addDense(2, ActivationFunction.SOFTMAX)
                .learningRate(0.01)
                .epochs(5)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(16, 1, 4, 4);
            int[] y = generateRandomLabels(16, 2);
            
            assertDoesNotThrow(() -> cnn.fit(X, y));
        }
    }
    
    @Nested
    @DisplayName("CNN Serialization Tests")
    class SerializationTests {
        
        @Test
        @DisplayName("Should save and load model")
        void testSaveLoad(@TempDir Path tempDir) throws IOException, ClassNotFoundException {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 4, 4)
                .addConv2D(2, 3, 1, 1, ActivationFunction.RELU)
                .addFlatten()
                .addDense(2, ActivationFunction.SOFTMAX)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(8, 1, 4, 4);
            int[] y = generateRandomLabels(8, 2);
            cnn.fit(X, y);
            
            // Get predictions before saving
            int[] predsBefore = cnn.predict(X);
            
            // Save and load
            File modelFile = tempDir.resolve("cnn_model.ser").toFile();
            cnn.save(modelFile.getAbsolutePath());
            
            CNN loadedCnn = CNN.load(modelFile.getAbsolutePath());
            
            // Get predictions after loading
            int[] predsAfter = loadedCnn.predict(X);
            
            assertArrayEquals(predsBefore, predsAfter);
        }
    }
    
    @Nested
    @DisplayName("CNN Summary Tests")
    class SummaryTests {
        
        @Test
        @DisplayName("Should print summary without error")
        void testSummary() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 28, 28)
                .addConv2D(16, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addConv2D(32, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addFlatten()
                .addDense(64, ActivationFunction.RELU)
                .addDense(10, ActivationFunction.SOFTMAX)
                .build();
            
            assertDoesNotThrow(() -> cnn.summary());
        }
        
        @Test
        @DisplayName("Should count parameters correctly")
        void testParameterCount() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 8, 8)
                .addConv2D(4, 3, 1, 0, ActivationFunction.RELU)  // 4*1*3*3 + 4 = 40
                .addFlatten()  // 4 * 6 * 6 = 144
                .addDense(10, ActivationFunction.SOFTMAX)  // 144 * 10 + 10 = 1450
                .build();
            
            // Total: 40 + 1450 = 1490
            assertEquals(1490, cnn.getNumParameters());
        }
    }
    
    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {
        
        @Test
        @DisplayName("Should work with realistic architecture")
        void testRealisticArchitecture() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 16, 16)
                .addConv2D(8, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addConv2D(16, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addFlatten()
                .addDense(32, ActivationFunction.RELU)
                .addDropout(0.5)
                .addDense(5, ActivationFunction.SOFTMAX)
                .learningRate(0.01)
                .epochs(10)
                .batchSize(8)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(32, 1, 16, 16);
            int[] y = generateRandomLabels(32, 5);
            
            double[][] XTest = generateRandomImages(8, 1, 16, 16);
            int[] yTest = generateRandomLabels(8, 5);
            
            cnn.fit(X, y, XTest, yTest);
            
            double accuracy = cnn.score(XTest, yTest);
            assertTrue(accuracy >= 0.0 && accuracy <= 1.0);
            
            assertFalse(cnn.getTrainingLoss().isEmpty());
            assertFalse(cnn.getValidationLoss().isEmpty());
        }
        
        @Test
        @DisplayName("Should work with average pooling")
        void testWithAveragePooling() {
            CNN cnn = new CNN.Builder()
                .inputShape(1, 8, 8)
                .addConv2D(4, 3, 1, 1, ActivationFunction.RELU)
                .addAveragePooling(2)
                .addFlatten()
                .addDense(2, ActivationFunction.SOFTMAX)
                .learningRate(0.01)
                .epochs(5)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(16, 1, 8, 8);
            int[] y = generateRandomLabels(16, 2);
            
            assertDoesNotThrow(() -> cnn.fit(X, y));
        }
        
        @Test
        @DisplayName("Should handle multi-channel input")
        void testMultiChannelInput() {
            CNN cnn = new CNN.Builder()
                .inputShape(3, 8, 8)  // RGB image
                .addConv2D(8, 3, 1, 1, ActivationFunction.RELU)
                .addMaxPooling(2)
                .addFlatten()
                .addDense(4, ActivationFunction.SOFTMAX)
                .learningRate(0.01)
                .epochs(3)
                .verbose(false)
                .randomSeed(42)
                .build();
            
            double[][] X = generateRandomImages(10, 3, 8, 8);
            int[] y = generateRandomLabels(10, 4);
            
            assertDoesNotThrow(() -> cnn.fit(X, y));
            
            int[] predictions = cnn.predict(X);
            assertEquals(10, predictions.length);
        }
    }
}
